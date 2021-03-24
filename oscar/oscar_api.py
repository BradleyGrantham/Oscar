from argparse import Namespace

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from run_captioning import load_models, get_captions, build_d2_model


class Image(BaseModel):
    b64string: str

MY_ARGS = Namespace(
    model_name_or_path="/home/ubuntu/Oscar/models/checkpoint-29-66420/",
    loss_type="sfmx",
    config_name="",
    tokenizer_name="",
    max_seq_length=70,
    max_seq_a_length=40,
    do_train=False,
    do_test=True,
    do_eval=True,
    do_lower_case=True,
    mask_prob=0.15,
    max_masked_tokens=3,
    add_od_labels=True,
    drop_out=0.1,
    max_img_seq_length=50,
    img_feature_dim=2054,
    img_feature_type="frcnn",
    per_gpu_train_batch_size=1,
    per_gpu_eval_batch_size=1,
    output_mode="classification",
    num_labels=2,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,
    weight_decay=0.05,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    warmup_steps=0,
    scheduler="linear",
    num_workers=4,
    num_train_epochs=40,
    max_steps=-1,
    logging_steps=20,
    save_steps=-1,
    evaluate_during_training=False,
    no_cuda=False,
    seed=88,
    scst=False,
    eval_model_dir="/home/ubuntu/Oscar/models/checkpoint-29-66420/",
    max_gen_length=20,
    output_hidden_states=False,
    num_return_sequences=1,
    num_beams=5,
    num_keep_best=8,
    temperature=1,
    top_k=0,
    top_p=1,
    repetition_penalty=1,
    length_penalty=1,
    use_cbs=False,
    min_constraints_to_satisfy=2
)

D2_MODEL = build_d2_model(not MY_ARGS.no_cuda)
MY_ARGS, MODEL, TOKENIZER = load_models(MY_ARGS)

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/oscar/")
def read_item(image: Image):
    b64string = image.b64string.split("base64,")[-1]
    captions = get_captions(MY_ARGS, b64string, MODEL, TOKENIZER, D2_MODEL)
    return captions
