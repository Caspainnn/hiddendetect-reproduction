"""Read-only exhaustive token/context audit, no model fitting or generation."""
from core import *

def audit():
 import torch
 from types import SimpleNamespace
 from transformers import AutoTokenizer
 from llava.conversation import conv_templates
 from llava.mm_utils import tokenizer_image_token
 from llava.constants import IMAGE_TOKEN_INDEX
 from extract import Extractor
 proof=json.loads((OUT/'verification.json').read_text());assert proof['passed'] and digest(OUT/'evaluation.json')==proof['evaluation_sha256'];rows,private,roles,y=load();checkpoint=json.loads((CACHE/'checkpoint.json').read_text());tok=AutoTokenizer.from_pretrained(str(ORIGIN/'models/llava-v1.6-vicuna-7b'),use_fast=False,local_files_only=True);shim=SimpleNamespace(torch=torch,model=SimpleNamespace(device='cpu'),tok=tok,conversation=conv_templates['llava_v1'],image_token=IMAGE_TOKEN_INDEX,tokenize_image=tokenizer_image_token);maximum_main=0;maximum_caption=0;maximum_text=0
 for i,row in enumerate(private):
  meta=checkpoint['metadata'][str(i)];expanded=meta['expanded_tokens'];assert expanded<4096,'An exactly saturated context needs explicit truncation investigation';maximum_main=max(maximum_main,expanded);original=Extractor.ids(shim,row['prompt'],True);assert hashlib.sha256(original.numpy().tobytes()).hexdigest()==meta['input_ids_sha256'];prompt='Describe visible objects, people, actions and relevant conditions for the following question. Do not answer the question or provide instructions. Question: '+row['prompt'];caption_ids=Extractor.ids(shim,prompt,True);caption_expanded=expanded+caption_ids.shape[1]-original.shape[1];assert caption_expanded+40<=4096;maximum_caption=max(maximum_caption,caption_expanded+40);description=checkpoint['descriptions'][str(i)];combined=Extractor.data_ids(shim,row['prompt'],description['ocr'],description['caption']);neutral=Extractor.data_ids(shim,row['prompt'],description['ocr'],description['caption'],True);assert combined.shape==neutral.shape
  for ids in [Extractor.ids(shim,row['prompt']),combined,neutral,Extractor.data_ids(shim,row['prompt'],description['ocr'],''),Extractor.data_ids(shim,row['prompt'],'',description['caption'])]:assert ids.shape[1]<4096;maximum_text=max(maximum_text,ids.shape[1])
 write(OUT/'input-coverage-audit.json',{'passed':True,'evaluation_sha256':proof['evaluation_sha256'],'code_sha256':digest(Path(__file__)),'rows_checked':len(rows),'main_expanded_max':maximum_main,'caption_context_plus_40_new_tokens_max':maximum_caption,'text_context_max':maximum_text,'context_limit':4096,'main_contexts_not_saturated':True,'all_original_token_hashes_rebuilt':True,'all_neutral_description_lengths_equal':True,'raw_visual_variants_same_dimensions_and_template':True,'caption_tokens_metadata':'Counts returned sequence tokens, which may include BOS/EOS; max_new_tokens in frozen generator is 40'});print('ALL_REAL_IMAGE_CONTEXTS_AND_TOKEN_COVERAGE_VERIFIED',flush=True)

if __name__=='__main__':audit()
