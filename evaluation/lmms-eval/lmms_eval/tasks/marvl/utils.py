import PIL.Image
from datasets import load_dataset, Image
import PIL

is_our_model = True

MARVL_RAW_IMAGE_DATASET = None
MARVL_ID2IMAGE = None

def marvl_doc_to_visual(doc):
    global MARVL_RAW_IMAGE_DATASET
    global MARVL_ID2IMAGE
    if MARVL_RAW_IMAGE_DATASET is None:
        MARVL_RAW_IMAGE_DATASET = load_dataset("floschne/marvl", token=True)
        MARVL_ID2IMAGE = {}
        for split in MARVL_RAW_IMAGE_DATASET:
            for row in MARVL_RAW_IMAGE_DATASET[split]:
                image = Image().decode_example(row["horizontally_stacked_img"])
                MARVL_ID2IMAGE[row["id"]] = image.convert("RGB")

    image = MARVL_ID2IMAGE[doc["id"]]
    return [image]

def nlvr2_doc_to_visual(doc):
    id = doc['id'].strip().lower()
    image = Image().decode_example(doc["image"]).convert("RGB")
    return [image]

def marvl_doc_to_text(doc, model_specific_prompt_kwargs):
    hypothesis = doc["hypothesis"].strip()
    #### This instruction is for our model
    if model_specific_prompt_kwargs['is_our_model']: instruction = f"Does the content of the image support the given text? Text: {hypothesis.strip()}\nOptions: (a) false (b) true" + '. Answer with the option directly.'
    #### mblip instruction
    # instruction = f'Based on the two images, is it correct to say "{hypothesis.strip()}"? Yes or no?'
    #### This instruction is for models other than our model
    else: instruction = f"Hypothesis: {hypothesis}\nIs it true that the hypothesis match the content of the image? Answer with one word from ['true', 'false']."
    return instruction

def nlvr2_doc_to_text(doc, model_specific_prompt_kwargs):
    conversations = doc['conversations']
    query = conversations[0]['value'].replace('<image>\n', '').strip() + '. Answer with the option directly.'
    if model_specific_prompt_kwargs['is_our_model']: return query
    # for models other than our model
    hypothesis = query.replace('Does the content of the image support the given text? Text: ', '').replace('Options: (a) false (b) true', '').strip()
    instruction = f"Hypothesis: {hypothesis}\nIs it true that the hypothesis match the content of the image? Answer with one word from ['true', 'false']."
    return instruction
    
def marvl_doc_to_target(doc):
    answer = doc["label"]
    answer = str(answer).lower()
    if 'true' in answer or 'True' in answer: return "true"
    else: return "false"

def nlvr2_doc_to_target(doc):
    conversations = doc['conversations']
    answer = str(conversations[1])
    if 'true' in answer: return 'true'
    elif 'false' in answer: return 'false'
    else: raise Exception(f"get target failed for id {doc['id']} - conversations: {conversations}")

def marvl_process_result(doc, results):
    target = marvl_doc_to_target(doc)
    # print(f"results[0]: {results[0]}")
    pred = results[0]
    pred = pred.strip().lower()
    #if 'no' in pred or '不' in pred or '否' in pred or '错' in pred or '錯' in pred: pred = 'false'
    #elif 'true' not in pred and 'false' not in pred: pred = 'true'
    if 'yes' in pred: pred = 'true'
    elif 'no' in pred: pred = 'false'
    if target.strip().lower() in pred:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}

def nlvr2_process_result(doc, results):
    target = nlvr2_doc_to_target(doc)
    pred = results[0]
    pred = pred.strip().lower()
    if 'yes' in pred: pred = 'true'
    elif 'no' in pred: pred = 'false'
    if target.strip().lower() in pred:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}
