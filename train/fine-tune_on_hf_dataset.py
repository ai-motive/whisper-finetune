import argparse
import multiprocessing
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tuning script for Whisper Models of various sizes.'
    )
    parser.add_argument(
        '--model_name', type=str, required=False,
        default='openai/whisper-small',
        help='Huggingface model name to fine-tune.'
    )
    parser.add_argument(
        '--language', type=str, required=False,
        default='Hindi',
        help='Language the model is being adapted to.'
    )
    parser.add_argument(
        '--sampling_rate', type=int, required=False,
        default=16000,
        help='Sampling rate of audio.'
    )
    parser.add_argument(
        '--num_proc', type=int, required=False,
        default=2,
        help='Number of parallel jobs for dataset operations.'
    )
    parser.add_argument(
        '--train_strategy', type=str, required=False,
        default='steps', choices=['steps', 'epoch'],
        help='Training strategy: steps or epoch.'
    )
    parser.add_argument(
        '--learning_rate', type=float, required=False,
        default=1.75e-5,
        help='Learning rate for fine-tuning.'
    )
    parser.add_argument(
        '--warmup', type=int, required=False,
        default=20000,
        help='Number of warmup steps.'
    )
    parser.add_argument(
        '--train_batchsize', type=int, required=False,
        default=48,
        help='Batch size during training.'
    )
    parser.add_argument(
        '--eval_batchsize', type=int, required=False,
        default=32,
        help='Batch size during evaluation.'
    )
    parser.add_argument(
        '--num_epochs', type=int, required=False,
        default=20,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--num_steps', type=int, required=False,
        default=100000,
        help='Number of steps to train.'
    )
    parser.add_argument(
        '--resume_from_ckpt', type=str, required=False,
        default=None,
        help='Path to checkpoint to resume from.'
    )
    parser.add_argument(
        '--output_dir', type=str, required=False,
        default='output_model_dir',
        help='Output directory for checkpoints.'
    )
    parser.add_argument(
        '--train_datasets', type=str, nargs='+', required=True,
        help='List of datasets for training.'
    )
    parser.add_argument(
        '--train_dataset_configs', type=str, nargs='+', required=True,
        help='Configs for each train dataset.'
    )
    parser.add_argument(
        '--train_dataset_splits', type=str, nargs='+', required=True,
        help='Split names for each train dataset.'
    )
    parser.add_argument(
        '--train_dataset_text_columns', type=str, nargs='+', required=True,
        help='Text column names for each train dataset.'
    )
    parser.add_argument(
        '--eval_datasets', type=str, nargs='+', required=True,
        help='List of datasets for evaluation.'
    )
    parser.add_argument(
        '--eval_dataset_configs', type=str, nargs='+', required=True,
        help='Configs for each eval dataset.'
    )
    parser.add_argument(
        '--eval_dataset_splits', type=str, nargs='+', required=True,
        help='Split names for each eval dataset.'
    )
    parser.add_argument(
        '--eval_dataset_text_columns', type=str, nargs='+', required=True,
        help='Text column names for each eval dataset.'
    )
    return parser.parse_args()


def load_all_datasets(args, split: str) -> Any:
    combined = []
    if split == 'train':
        iterable = zip(
            args.train_datasets,
            args.train_dataset_configs,
            args.train_dataset_splits,
            args.train_dataset_text_columns
        )
    else:
        iterable = zip(
            args.eval_datasets,
            args.eval_dataset_configs,
            args.eval_dataset_splits,
            args.eval_dataset_text_columns
        )

    for ds_name, config, split_name, text_col in iterable:
        ds = load_dataset(ds_name, config, split=split_name)
        ds = ds.cast_column("audio", Audio(args.sampling_rate))
        if text_col != "sentence":
            ds = ds.rename_column(text_col, "sentence")
        ds = ds.remove_columns(
            set(ds.features.keys()) - {"audio", "sentence"}
        )
        combined.append(ds)

    return concatenate_datasets(combined).shuffle(seed=22)


def prepare_dataset(
    batch: Dict[str, Any],
    processor: WhisperProcessor,
    do_lower_case: bool,
    do_remove_punct: bool,
    normalizer: BasicTextNormalizer
) -> Dict[str, Any]:
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punct:
        transcription = normalizer(transcription).strip()

    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch


def is_in_length_range(
    length: float,
    labels: List[int],
    max_label_length: int
) -> bool:
    return 0.0 < length < 30.0 and 0 < len(labels) < max_label_length


def filter_by_length(example: Dict[str, Any], max_label_length: int) -> bool:
    return is_in_length_range(
        example["input_length"], example["labels"], max_label_length
    )

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_feats, return_tensors="pt"
        )

        label_feats = [{"input_ids": f["labels"]} for f in features]
        label_batch = self.processor.tokenizer.pad(
            label_feats, return_tensors="pt"
        )
        labels = label_batch["input_ids"].masked_fill(
            label_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def compute_metrics(
    pred, processor: WhisperProcessor,
    normalizer: BasicTextNormalizer,
    metric, do_normalize_eval: bool
) -> Dict[str, float]:
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True
    )
    label_str = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True
    )
    if do_normalize_eval:
        pred_str = [normalizer(p) for p in pred_str]
        label_str = [normalizer(l) for l in label_str]
    wer = 100 * metric.compute(
        predictions=pred_str, references=label_str
    )
    return {"wer": wer}


def main():
    args = parse_args()

    # 검증
    if len(args.train_datasets) == 0 or len(args.eval_datasets) == 0:
        raise ValueError("학습 및 평가용 데이터셋을 지정해주세요.")

    # 프로세서 및 모델 로드
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    max_label_length = model.config.max_length

    if model.config.decoder_start_token_id is None:
        raise ValueError("decoder_start_token_id이 정의되지 않았습니다.")

    # 옵션 설정
    gradient_checkpointing = True
    if gradient_checkpointing:
        model.config.use_cache = False

    # 데이터셋 로딩
    print("▶ 데이터셋 로딩 중...")
    raw_ds = DatasetDict({
        "train": load_all_datasets(args, "train"),
        "eval": load_all_datasets(args, "eval")
    })

    # 전처리
    print("▶ 전처리(map) 실행...")
    raw_ds = raw_ds.map(
        lambda b: prepare_dataset(
            b, processor,
            do_lower_case=False,
            do_remove_punct=False,
            normalizer=BasicTextNormalizer()
        ),
        num_proc=args.num_proc,
        remove_columns=["audio", "sentence"]
    )

    # 필터링 (멀티프로세싱)
    print("▶ 필터링 실행...")
    raw_ds = raw_ds.filter(
        filter_by_length,
        fn_kwargs={"max_label_length": max_label_length},
        num_proc=args.num_proc
    )

    # DataCollator 및 Metric
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    wer_metric = evaluate.load("wer")

    # TrainingArguments 설정
    if args.train_strategy == 'epoch':
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.train_batchsize,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=args.num_epochs,
            warmup_steps=args.warmup,
            learning_rate=args.learning_rate,
            gradient_checkpointing=gradient_checkpointing,
            fp16=True,
            save_total_limit=10,
            per_device_eval_batch_size=args.eval_batchsize,
            predict_with_generate=True,
            logging_steps=500,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            optim="adamw_bnb_8bit",
            resume_from_checkpoint=args.resume_from_ckpt
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.train_batchsize,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            max_steps=args.num_steps,
            warmup_steps=args.warmup,
            learning_rate=args.learning_rate,
            gradient_checkpointing=gradient_checkpointing,
            fp16=True,
            save_total_limit=10,
            per_device_eval_batch_size=args.eval_batchsize,
            predict_with_generate=True,
            logging_steps=500,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            optim="adamw_bnb_8bit",
            resume_from_checkpoint=args.resume_from_ckpt
        )

    # Trainer 초기화 및 학습
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_ds["train"],
        eval_dataset=raw_ds["eval"],
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(
            p, processor, BasicTextNormalizer(), wer_metric, True
        ),
        tokenizer=processor.tokenizer
    )

    processor.save_pretrained(args.output_dir)
    print("▶ 학습 시작...")
    trainer.train()
    print("✔ 학습 완료")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
