from together import Together
import os
from dotenv import load_dotenv
import json
import argparse
from together.utils import check_file


def check_dataset(file_path: str):
    sft_report = check_file(file_path)
    print(json.dumps(sft_report, indent=2))
    assert sft_report['is_check_passed']
    return sft_report

def upload_file(client: Together, file_path: str):
    resp = client.files.upload(file_path, check=True)
    print(f"Uploaded file ID: {resp.id}")
    print("Save this ID for starting your fine-tuning job.")
    return resp


def finetune_model(client: Together, file_id: str, model_name: str, wandb_api_key: str, suffix: str, epochs: int):
    ft_resp = client.fine_tuning.create(
        training_file=file_id,
        model=model_name,
        train_on_inputs=False,
        n_epochs=epochs,
        n_checkpoints=epochs,
        wandb_api_key=wandb_api_key,  # Optional, for visualization
        lora=True,  # Default True
        warmup_ratio=0,
        learning_rate=1e-5,
        suffix=suffix,
    )
    print(f"Fine-tuning job ID: {ft_resp.id}")  # Save this job ID for monitoring
    print(f"Save this for monitoring your fine-tuning job.")
    return ft_resp


def download_model(client: Together, ft_job_id: str):
    model_resp = client.fine_tuning.retrieve(ft_job_id)
    model_file = client.fine_tuning.download(ft_job_id, 
                                            #  checkpoint_type='adapter'
                                             )
    print(f"Model downloaded to: {model_file}")
    return model_file

def get_ft_model_name(client: Together, ft_job_id: str):
    model_resp = client.fine_tuning.retrieve(ft_job_id)
    return model_resp.output_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=False, default="data/sft_data.jsonl")
    parser.add_argument('--model_name', type=str, required=False, default="Llama-3.1-8B-Instruct-Reference")
    parser.add_argument('--suffix', type=str, required=False, default="finetuned_model")
    parser.add_argument('--check_dataset', action='store_true')
    parser.add_argument('--upload_file', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--file_id', type=str, required=False, default=None)
    parser.add_argument('--ft_job_id', type=str, required=False, default=None)
    parser.add_argument('--download_model', action='store_true')
    parser.add_argument('--get_model_name', action='store_true')
    parser.add_argument('--epochs', type=int, required=False, default=10)
    args = parser.parse_args()

    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    WANDB_API_KEY = os.getenv(
        "WANDB_API_KEY"
    ) 
    client = Together(api_key=TOGETHER_API_KEY)

    if args.check_dataset:
        check_dataset(args.file_path)
    if args.upload_file:
        upload_file(client, args.file_path)
    if args.finetune:
        if args.file_id is not None:
            file_id = args.file_id
            finetune_model(client, file_id, args.model_name, WANDB_API_KEY, args.suffix, args.epochs)
        else:
            assert args.file_id is not None, "File ID must be provided for fine-tuning."
    if args.download_model:
        if args.ft_job_id is not None:
            download_model(client, args.ft_job_id)
        else:
            assert args.ft_job_id is not None, "Fine-tuning job ID must be provided for downloading the model."
    if args.get_model_name:
        if args.ft_job_id is not None:
            model_name = get_ft_model_name(client, args.ft_job_id)
            print(f"Fine-tuned model name: {model_name}")
        else:
            assert args.ft_job_id is not None, "Fine-tuning job ID must be provided for getting the model name."
         