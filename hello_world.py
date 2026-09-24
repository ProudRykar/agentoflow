from textual.app import App
from textual.widgets import Label, Button

class HelloWorld(App):
    """A simple hello world app."""

    def compose(self) -> list:
        yield Label("Hello, World!")
        yield Button("Exit")

    def on_button_pressed(self) -> None:
        """Handle the button press to exit the application."""
        self.exit()

if __name__ == "__main__":
    app = HelloWorld()
    app.run()
