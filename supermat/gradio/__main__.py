from supermat.gradio import create_llm_interface

demo = create_llm_interface()
demo.launch(
    share=False,
    debug=True,
)
