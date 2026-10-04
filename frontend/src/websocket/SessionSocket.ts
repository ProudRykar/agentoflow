import { socketUrl } from '../api/client'
import type { WireEnvelope } from '../api/types'

export interface SocketHandlers {
  onEvent: (event: WireEnvelope) => void
  onStatus: (status: 'connecting' | 'open' | 'closed', error?: string | null) => void
}

const BASE_DELAY_MS = 500
const MAX_DELAY_MS = 10_000

/**
 * Auto-reconnecting session socket.
 *
 * The cursor is the highest sequence number the client has
 * processed. On every (re)connect it is sent as `since`, so the
 * backend replays exactly what was missed instead of the client
 * polling for state.
 */
export class SessionSocket {
  private socket: WebSocket | null = null
  private closed = false
  private attempt = 0
  private timer: number | null = null
  private pingTimer: number | null = null

  constructor(
    private readonly sessionId: string,
    private cursor: number,
    private readonly handlers: SocketHandlers,
  ) {}

  get lastSeq(): number {
    return this.cursor
  }

  connect(): void {
    if (this.closed) {
      return
    }

    this.handlers.onStatus('connecting')

    const socket = new WebSocket(socketUrl(this.sessionId, this.cursor))

    this.socket = socket

    socket.onopen = () => {
      this.attempt = 0
      this.handlers.onStatus('open', null)
      this.startPing()
    }

    socket.onmessage = (message) => {
      const text =
        typeof message.data === 'string' ? message.data : ''

      if (text === 'pong') {
        return
      }

      let parsed: WireEnvelope

      try {
        parsed = JSON.parse(text) as WireEnvelope
      } catch {
        return
      }

      if (typeof parsed.seq === 'number') {
        // Ignore duplicates that a replay may re-deliver.
        if (parsed.seq <= this.cursor) {
          return
        }

        this.cursor = parsed.seq
      }

      this.handlers.onEvent(parsed)
    }

    socket.onerror = () => {
      socket.close()
    }

    socket.onclose = () => {
      this.stopPing()
      this.socket = null

      if (this.closed) {
        this.handlers.onStatus('closed')
        return
      }

      this.handlers.onStatus('closed', 'connection lost')
      this.scheduleReconnect()
    }
  }

  /** Send a keepalive so proxies do not silently drop the socket. */
  private startPing(): void {
    this.stopPing()

    this.pingTimer = window.setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.socket.send('ping')
      }
    }, 15_000)
  }

  private stopPing(): void {
    if (this.pingTimer !== null) {
      window.clearInterval(this.pingTimer)
      this.pingTimer = null
    }
  }

  private scheduleReconnect(): void {
    this.attempt += 1

    const delay = Math.min(
      BASE_DELAY_MS * 2 ** (this.attempt - 1),
      MAX_DELAY_MS,
    )

    this.timer = window.setTimeout(() => {
      this.timer = null
      this.connect()
    }, delay)
  }

  close(): void {
    this.closed = true

    if (this.timer !== null) {
      window.clearTimeout(this.timer)
      this.timer = null
    }

    this.stopPing()

    this.socket?.close()
    this.socket = null
  }
}
