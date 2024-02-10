import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from rouge import Rouge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define dataclass for evaluation arguments
@dataclass
class EvalArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    valid_file_path: str = field(metadata={"help": "Path for cached valid dataset"})
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    tokenizer_name_or_path: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    num_beams: Optional[int] = field(default=4, metadata={"help": "num_beams to use for decoding"})
    max_decoding_length: Optional[int] = field(default=32, metadata={"help": "maximum length for decoding"})
    output_path: Optional[str] = field(default="hypothesis.txt", metadata={"help": "path to save the generated questions."})
    save_model_path: Optional[str] = field(default="best_model", metadata={"help": "path to save the best model"})

# Function to compute BLEU score
def compute_bleu(predictions, references):
    return corpus_bleu(references, predictions)

# Function to compute ROUGE score
def compute_rouge(predictions, references):
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

# Function to compute METEOR score
def compute_meteor(predictions, references):
    scores = [meteor_score([ref], pred) for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores)

# Function to compute BERTScore
def compute_bertscore(predictions, references):
    _, _, f1 = bert_score(predictions, references, lang='en', verbose=False)
    return f1.mean().item()

# Function to get model predictions
def get_predictions(model, tokenizer, data_loader, num_beams=4, max_length=32, length_penalty=1):
    device = next(model.parameters()).device
    model.to(device)
    predictions = []
    model.eval()
    with torch.no_grad(), tqdm(total=len(data_loader)) as progress_bar:
        for batch in data_loader:
            outs = model.generate(
                input_ids=batch['input_ids'].to(device), 
                attention_mask=batch['attention_mask'].to(device),
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
            )
            prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            predictions.extend(prediction)
            progress_bar.update(1)
    return predictions

# Main function
def main():
    # Parse arguments
    parser = HfArgumentParser((EvalArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path or args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # Load validation dataset and create data loader
    valid_dataset = torch.load(args.valid_file_path)
    collator = T2TDataCollator(tokenizer=tokenizer, model_type=args.model_type, mode="inference")
    loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, collate_fn=collator)

    # Generate predictions
    predictions = get_predictions(
        model=model,
        tokenizer=tokenizer,
        data_loader=loader,
        num_beams=args.num_beams,
        max_length=args.max_decoding_length
    )

    # Compute evaluation metrics
    bleu_score = compute_bleu(predictions, valid_dataset['references'])
    rouge1_score, rouge2_score, rougel_score = compute_rouge(predictions, valid_dataset['references'])
    meteor_score = compute_meteor(predictions, valid_dataset['references'])
    bert_score = compute_bertscore(predictions, valid_dataset['references'])

    # Log and print performance metrics
    logger.info("Performance Metrics:")
    logger.info(f"BLEU Score: {bleu_score:.4f}")
    logger.info(f"ROUGE-1 Score: {rouge1_score:.4f}")
    logger.info(f"ROUGE-2 Score: {rouge2_score:.4f}")
    logger.info(f"ROUGE-L Score: {rougel_score:.4f}")
    logger.info(f"METEOR Score: {meteor_score:.4f}")
    logger.info(f"BERTScore F1: {bert_score:.4f}")

    # Determine the best score achieved
    best_metric_score = max(bleu_score, rouge1_score, rouge2_score, rougel_score, meteor_score, bert_score)
    best_metric = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'BERTScore F1']
    best_metric_idx = [bleu_score, rouge1_score, rouge2_score, rougel_score, meteor_score, bert_score].index(best_metric_score)
    logger.info(f"Best score achieved on: {best_metric[best_metric_idx]} - {best_metric_score:.4f}")

    # Save the model with the best score
    if args.save_model_path:
        model.save_pretrained(args.save_model_path)

    # Save predictions to output file
    with open(args.output_path, 'w') as f:
        f.write("\n".join(predictions))
    logger.info(f"Output saved at {args.output_path}")

if __name__ == "__main__":
    main()
