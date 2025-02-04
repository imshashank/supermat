from enum import StrEnum
from typing import TYPE_CHECKING

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from pydantic import SecretStr


class LLMProvider(StrEnum):
    ollama = "Ollama"
    anthropic = "Anthropic"
    openai = "OpenAI"


class LLMChat:
    def __init__(self):
        self.chat_model = None
        self.provider = None
        self.model = None

    def initialize_client(
        self,
        provider: str,
        model: str,
        credentials: str | None = None,
        base_url: str | None = None,
        temperature: float | None = 0.0,
    ) -> str:
        """Initialize the LangChain chat model based on selected provider."""
        self.provider = provider
        self.model = model

        if TYPE_CHECKING:
            assert isinstance(credentials, SecretStr)

        try:
            match (provider):
                case LLMProvider.ollama:
                    from langchain_ollama.llms import OllamaLLM  # noqa: I900

                    self.chat_model = OllamaLLM(model=model, temperature=temperature, base_url=base_url)
                case LLMProvider.anthropic:
                    from langchain_anthropic import ChatAnthropic  # noqa: I900

                    self.chat_model = ChatAnthropic(
                        model_name=model, temperature=temperature, timeout=None, api_key=credentials, stop=None
                    )
                case LLMProvider.openai:
                    from langchain_openai import ChatOpenAI  # noqa: I900

                    temperature = 0.7 if temperature is None else temperature
                    self.chat_model = ChatOpenAI(model=model, temperature=temperature, api_key=credentials)
                case _:
                    return f"Invalid LLM Provider {provider}"

            return f"{self.chat_model.get_name()} initialized successfully!"

        except Exception as e:
            return f"Error initializing client: {str(e)}"

    def convert_history_to_messages(self, history: list[dict]) -> list[HumanMessage | AIMessage]:
        """Convert Gradio chat history to LangChain message format."""

        return [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in history
        ]

    def chat(self, message: str, history):
        """Process chat message using LangChain chat model."""
        if not self.chat_model:
            return "Please initialize an LLM provider first!"

        try:
            history_langchain_format = self.convert_history_to_messages(history)
            history_langchain_format.append(HumanMessage(content=message))
            gpt_response = self.chat_model.invoke(history_langchain_format)
            return gpt_response if isinstance(gpt_response, str) else gpt_response.content

        except Exception as e:
            return f"Error: {str(e)}"


def create_llm_interface():
    llm_chat = LLMChat()

    with gr.Blocks() as app:
        with gr.Tab("Setup"):
            with gr.Row():
                provider_option = gr.Dropdown(
                    choices=["--SELECT--"] + list(map(str, LLMProvider)), label="Select LLM Provider"
                )

            @gr.render(inputs=provider_option, triggers=[provider_option.change])
            def provider_update(provider: str):
                if provider == "--SELECT--":
                    gr.Error("Select a provider from dropdown.", print_exception=False)
                    return
                with gr.Row():
                    llm_init_inputs = []
                    initialize_client = llm_chat.initialize_client
                    with gr.Column():
                        if provider == LLMProvider.ollama:
                            base_url = gr.Textbox(
                                value="http://localhost:11434",
                                label="API Host",
                                placeholder="e.g., http://localhost:11434 for Ollama",
                            )
                            model = gr.Textbox(
                                value="deepseek-r1:8b", label="Model Name", placeholder="Model setup in Ollama"
                            )
                            llm_init_inputs = [provider_option, model, base_url]
                            initialize_client = lambda p, m, b: llm_chat.initialize_client(  # noqa: E731
                                provider=p,
                                model=m,
                                base_url=b,
                            )
                        else:
                            credentials = gr.Textbox(
                                label="API Key/Credentials", type="password", placeholder="Enter your API key here"
                            )
                            model = gr.Textbox(
                                label=f"Select {provider} model",
                            )
                            llm_init_inputs = [provider_option, model, credentials]
                            initialize_client = lambda p, m, c: llm_chat.initialize_client(  # noqa: E731
                                provider=p, model=m, credentials=c
                            )

                    with gr.Column():
                        initialize_btn = gr.Button("Initialize LLM")
                        init_status = gr.Textbox(label="Initialization Status")

                        initialize_btn.click(fn=initialize_client, inputs=llm_init_inputs, outputs=init_status)

        with gr.Tab("Chat"):

            def respond(message, chat_history):
                bot_message = llm_chat.chat(message, chat_history)
                return bot_message

            gr.ChatInterface(respond, type="messages")

    return app
