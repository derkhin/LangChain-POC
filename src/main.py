import os
import streamlit as st
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


@tool
def ApparelGeneration(image) -> str:
    """Tool to create a design and print with text description and preview it on apparel. And for Generate Clothes and  for new clothes designs use this tool"""
    return "ApparelGenrationOutput"

@tool
def AddBackground(image) -> str:
    """To to Add or Remove things from Background Image and for other backgrond manipulation Use This Tool"""
    return "BackgroundManipulationOutput"

@tool
def FaceSwap(image) -> str:
    """Use this tool when the user is asking to manipulate or change a face in an image."""
    return "FaceSwapOutput"

tools = [ApparelGeneration, AddBackground, FaceSwap]
model_with_tools = llm.bind_tools(tools)
tools_map={tool.name: tool for tool in tools}


def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()

    prev_output = None
        
    for tool_call in tool_calls:
        if prev_output is not None:
            tool_call["args"] = prev_output
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        prev_output = tool_call["output"]
    return tool_calls


chain = model_with_tools | call_tools

if __name__ == '__main__':
    # set title
    st.title('Ask a question to an image')

    # set header
    st.header("Please upload an image")

    # upload file
    file = st.file_uploader("", type=["jpeg", "jpg", "png"])

    if file:
        # display image
        st.image(file, use_column_width=True)

        # text input
        user_question = st.text_input('Ask a question about your image:')

        with NamedTemporaryFile(dir='.') as f:
            f.write(file.getbuffer())
            image_path = 'C:/Users/Irfan/Desktop/test.jpg'
            # write agent response
            if user_question and user_question != "":
                with st.spinner(text="In progress..."):
                    response=chain.invoke([HumanMessage(content=f'{user_question}, and this is the image path: {image_path}')])
                    # response=chain.invoke([HumanMessage(content="please enhance my face to resemble a movie star and remove background from this image and  design a new t-shirt graphic")])
                    for i, step in enumerate(response, 1):
                        if i == 1:
                            st.write(f"function #{i} : {step['name']}({image_path})")
                        else:
                            st.write(f"function #{i} : {step['name']}({step['args']})")
                        st.write(f"Output #{i} : {step['output']}\n")
                    st.write(f"FinalOutput - {response[-1]['output']}")