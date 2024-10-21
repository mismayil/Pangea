from datasets import load_dataset

XGQA_RAW_IMAGE_DATASET = None
XGQA_ID2IMAGE = None


def xgqa_doc_to_visual(doc):
    global XGQA_RAW_IMAGE_DATASET
    global XGQA_ID2IMAGE
    if XGQA_RAW_IMAGE_DATASET is None:
        XGQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
        XGQA_ID2IMAGE = {}
        for row in XGQA_RAW_IMAGE_DATASET:
            XGQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = XGQA_ID2IMAGE[doc["image_id"]]
    return [image]


def xgqa_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"].strip()
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question} {post_prompt.strip()}"

def xgqa_process_result(doc, results):
    target = doc['answer'].strip().lower()
    # print(f"results[0]: {results[0]}")
    pred = results[0]
    pred = pred.strip().lower()
    if target in pred:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}