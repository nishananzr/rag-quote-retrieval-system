---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:2508
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: “I am very interested and fascinated how everyone loves each other,
    but no one really likes each other.”
  sentences:
  - 'Quote: "“Bran thought about it. ''Can a man still be brave if he''s afraid?''''That
    is the only time a man can be brave,'' his father told him.”" Author: George R.R.
    Martin,. Tags: fear, courage, bravery'
  - 'Quote: "“Stepping onto a brand-new path is difficult, but not more difficult
    than remaining in a situation, which is not nurturing to the whole woman.”" Author:
    Maya Angelou. Tags: change, inspiration, self determination'
  - 'Quote: "“I am very interested and fascinated how everyone loves each other, but
    no one really likes each other.”" Author: Stephen Chbosky,. Tags: moi'
- source_sentence: “If you remember me, then I don't care if everyone else forgets.”
  sentences:
  - 'Quote: "“If you remember me, then I don''t care if everyone else forgets.”" Author:
    Haruki Murakami,. Tags: inspiration, love'
  - 'Quote: "“you can, you should, and if youâ€™re brave enough to start, you will.”"
    Author: Stephen King,. Tags: writing, positive thinking, self empowerment, bravery'
  - 'Quote: "“Angry people are not always wise.”" Author: Jane Austen,. Tags: anger,
    jane austen, wisdom'
- source_sentence: “We are all born mad. Some remain so.”
  sentences:
  - 'Quote: "“Don''t compromise yourself - you''re all you have.”" Author: John Grisham,.
    Tags: be yourself, crime, courtroom drama'
  - 'Quote: "“We are all born mad. Some remain so.”" Author: Samuel Beckett. Tags:
    samuel beckett, rena silverman'
  - 'Quote: "“You know what charm is: a way of getting the answer yes without having
    asked any clear question.”" Author: Albert Camus,. Tags: romance'
- source_sentence: “Friendship ... is born at the moment when one man says to another
    "What! You too? I thought that no one but myself . . .”
  sentences:
  - 'Quote: "“Growing apart doesn''t change the fact that for a long time we grew
    side by side; our roots will always be tangled. I''m glad for that.”" Author:
    Ally Condie,. Tags: childhood, growing up, friendship'
  - 'Quote: "“It is said that your life flashes before your eyes just before you die.
    That is true, it''s called Life.”" Author: Terry Pratchett,. Tags: death, humor,
    life'
  - 'Quote: "“Friendship ... is born at the moment when one man says to another "What!
    You too? I thought that no one but myself . . .”" Author: C.S. Lewis,. Tags: friendship'
- source_sentence: “Thinking something does not make it true. Wanting something does
    not make it real.”
  sentences:
  - 'Quote: "“When I discover who I am, Iâ€™ll be free.”" Author: Ralph Ellison,.
    Tags: identity, self awareness, self discovery, independence, freedom'
  - 'Quote: "“Have you ever noticed that anybody driving slower than you is an idiot,
    and anyone going faster than you is a maniac?”" Author: George Carlin. Tags: humor'
  - 'Quote: "“Thinking something does not make it true. Wanting something does not
    make it real.”" Author: Michelle Hodkin,. Tags: inspiration, mara dyer, reality,
    truth'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '“Thinking something does not make it true. Wanting something does not make it real.”',
    'Quote: "“Thinking something does not make it true. Wanting something does not make it real.”" Author: Michelle Hodkin,. Tags: inspiration, mara dyer, reality, truth',
    'Quote: "“When I discover who I am, Iâ€™ll be free.”" Author: Ralph Ellison,. Tags: identity, self awareness, self discovery, independence, freedom',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2,508 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 8 tokens</li><li>mean: 42.09 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 61.19 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>“If heâ€™s not calling you, itâ€™s because you are not on his mind. If he creates expectations for you, and then doesnâ€™t follow through on little things, he will do same for big things. Be aware of this and realize that heâ€™s okay with disappointing you. Donâ€™t be with someone who doesnâ€™t do what they say theyâ€™re going to do. If heâ€™s choosing not to make a simple effort that would put you at ease and bring harmony to a recurring fight, then he doesnâ€™t respect your feelings and needs. â€œBusyâ€� is another word for â€œasshole.â€� â€œAssholeâ€� is another word for the guy youâ€™re dating. You deserve a fcking phone call.”</code> | <code>Quote: "“If heâ€™s not calling you, itâ€™s because you are not on his mind. If he creates expectations for you, and then doesnâ€™t follow through on little things, he will do same for big things. Be aware of this and realize that heâ€™s okay with disappointing you. Donâ€™t be with someone who doesnâ€™t do what they say theyâ€™re going to do. If heâ€™s choosing not to make a simple effort that would put you at ease and bring harmony to a recurring fight, then he doesnâ€™t respect your feelings and needs. â€œBusyâ€� is another word for â€œasshole.â€� â€œAssholeâ€� is another word for the guy youâ€™re dating. You deserve a fcking phone call.”" Author: Greg Behrendt. Tags: true, dating, faith, romance, marriage, he s not just into you, guys, busy, greg behrendt, dating advice, call, love</code> |
  | <code>“To love at all is to be vulnerable. Love anything and your heart will be wrung and possibly broken. If you want to make sure of keeping it intact you must give it to no one, not even an animal. Wrap it carefully round with hobbies and little luxuries; avoid all entanglements. Lock it up safe in the casket or coffin of your selfishness. But in that casket, safe, dark, motionless, airless, it will change. It will not be broken; it will become unbreakable, impenetrable, irredeemable. To love is to be vulnerable.”</code>                                                                                                                            | <code>Quote: "“To love at all is to be vulnerable. Love anything and your heart will be wrung and possibly broken. If you want to make sure of keeping it intact you must give it to no one, not even an animal. Wrap it carefully round with hobbies and little luxuries; avoid all entanglements. Lock it up safe in the casket or coffin of your selfishness. But in that casket, safe, dark, motionless, airless, it will change. It will not be broken; it will become unbreakable, impenetrable, irredeemable. To love is to be vulnerable.”" Author: C.S. Lewis,. Tags: love</code>                                                                                                                                                                                                                                              |
  | <code>“Time is an illusion. Lunchtime doubly so.”</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | <code>Quote: "“Time is an illusion. Lunchtime doubly so.”" Author: Douglas Adams,. Tags: science fiction, philosophy, humor</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.12.4
- Sentence Transformers: 4.1.0
- Transformers: 4.52.4
- PyTorch: 2.6.0+cpu
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->