export interface PendingApprovalView {
  approvalId: string
  toolName: string
  permission: string
  reason: string
  arguments: Record<string, unknown>
}
