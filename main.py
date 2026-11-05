#fastapi
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="T5 Text Summarization App", description="Text Summarization using T5 model", version="1.0.0")


# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_T5_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_T5_summary_model")


# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model.to(device)


# templating
templates = Jinja2Templates(directory=".")

# Input Schema
class Input_Schema(BaseModel):
    dialogue: str


def clean_data(text):
    text = re.sub(r"\r\n", " ", text) #replacing next-line component from text with empty space.
    text = re.sub(r"\s+", " ", text) #spaces
    text = re.sub(r"<.*?>", " ", text) #<p>, <h1> etc.
    text = text.strip().lower() #to remove trailing spaces and convert everything to lower case.
    return text


#the whole logic of text summarization in a function so that we can server it via an api.
def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) #clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    #generate the summary but it'll be first generated in the form of token ids.
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    # convert token ids to text => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary


#API endpoint for text summarization
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/summarize")
async def summarize(user_input: Input_Schema):
    summary = summarize_dialogue(user_input.dialogue)
    return {"summary": summary}

