# ML Papers Explained

Explanations to key concepts in ML

## Language Models

| Paper | Date | Description |
|---|---|---|
| [Transformer](https://ritvik19.medium.com/papers-explained-01-transformer-474bb60a33f7) | June 2017 | An Encoder Decoder model, that introduced multihead attention mechanism for language translation task. |
| [Elmo](https://ritvik19.medium.com/papers-explained-33-elmo-76362a43e4) | February 2018 | Deep contextualized word representations that captures both intricate aspects of word usage and contextual variations across language contexts. |
| [Marian MT](https://ritvik19.medium.com/papers-explained-150-marianmt-1b44479b0fd9) | April 2018 | A Neural Machine Translation framework written entirely in C++ with minimal dependencies, designed for high training and translation speed. |
| [GPT](https://ritvik19.medium.com/papers-explained-43-gpt-30b6f1e6d226) | June 2018 | A Decoder only transformer which is autoregressively pretrained and then finetuned for specific downstream tasks using task-aware input transformations. |
| [BERT](https://ritvik19.medium.com/papers-explained-02-bert-31e59abc0615) | October 2018 | Introduced pre-training for Encoder Transformers. Uses unified architecture across different tasks. |
| [Transformer XL](https://ritvik19.medium.com/papers-explained-34-transformerxl-2e407e780e8) | January 2019 | Extends the original Transformer model to handle longer sequences of text by introducing recurrence into the self-attention mechanism. |
| [XLM](https://ritvik19.medium.com/papers-explained-158-xlm-42a175e93caf) | January 2019 | Proposes two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual data, and one supervised that leverages parallel data with a new cross-lingual language model objective.  |
| [GPT 2](https://ritvik19.medium.com/papers-explained-65-gpt-2-98d0a642e520) | February 2019 | Demonstrates that language models begin to learn various language processing tasks without any explicit supervision. | 
| [Sparse Transformer](https://ritvik19.medium.com/papers-explained-122-sparse-transformer-906a0be1e4e7) | April 2019 | Introduced sparse factorizations of the attention matrix to reduce the time and memory consumption to O(n√ n) in terms of sequence lengths. |
| [UniLM](https://ritvik19.medium.com/papers-explained-72-unilm-672f0ecc6a4a) | May 2019 | Utilizes a shared Transformer network and specific self-attention masks to excel in both language understanding and generation tasks. |
| [XLNet](https://ritvik19.medium.com/papers-explained-35-xlnet-ea0c3af96d49) | June 2019 | Extension of the Transformer-XL, pre-trained using a new method that combines ideas from AR and AE objectives. |
| [RoBERTa](https://ritvik19.medium.com/papers-explained-03-roberta-81db014e35b9) | July 2019 | Built upon BERT, by carefully optimizing hyperparameters and training data size to improve performance on various language tasks . |
| [Sentence BERT](https://ritvik19.medium.com/papers-explained-04-sentence-bert-5159b8e07f21) | August 2019 | A modification of BERT that uses siamese and triplet network structures to derive sentence embeddings that can be compared using cosine-similarity. |
| [CTRL](https://ritvik19.medium.com/papers-explained-153-ctrl-146fcd18a566) | September 2019 | A 1.63B language model that can generate text conditioned on control codes that govern style, content, and task-specific behavior, allowing for more explicit control over text generation. |
| [Tiny BERT](https://ritvik19.medium.com/papers-explained-05-tiny-bert-5e36fe0ee173) | September 2019 | Uses attention transfer, and task specific distillation for distilling BERT. |
| [ALBERT](https://ritvik19.medium.com/papers-explained-07-albert-46a2a0563693) | September 2019 | Presents certain parameter reduction techniques to lower memory consumption and increase the training speed of BERT. |
| [Distil BERT](https://ritvik19.medium.com/papers-explained-06-distil-bert-6f138849f871) | October 2019 | Distills BERT on very large batches leveraging gradient accumulation, using dynamic masking and without the next sentence prediction objective. |
| [T5](https://ritvik19.medium.com/papers-explained-44-t5-9d974a3b7957) | October 2019 | A unified encoder-decoder framework that converts all text-based language problems into a text-to-text format. |
| [BART](https://ritvik19.medium.com/papers-explained-09-bart-7f56138175bd) | October 2019 | An Encoder-Decoder pretrained to reconstruct the original text from corrupted versions of it. |
| [XLM-Roberta](https://ritvik19.medium.com/papers-explained-159-xlm-roberta-2da91fc24059) | November 2019 | A multilingual masked language model pre-trained on text in 100 languages, shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of crosslingual transfer tasks. |
| [XLM-Roberta](https://ritvik19.medium.com/papers-explained-159-xlm-roberta-2da91fc24059) | November 2019 | A multilingual masked language model pre-trained on text in 100 languages, shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of crosslingual transfer tasks. |
| [Pegasus](https://ritvik19.medium.com/papers-explained-162-pegasus-1cb16f572553) | December 2019 | A self-supervised pre-training objective for abstractive text summarization, proposes removing/masking important sentences from an input document and generating them together as one output sequence. |
| [Reformer](https://ritvik19.medium.com/papers-explained-165-reformer-4445ad305191) | January 2020 | Improves the efficiency of Transformers by replacing dot-product attention with locality-sensitive hashing (O(Llog L) complexity), using reversible residual layers to store activations only once, and splitting feed-forward layer activations into chunks, allowing it to perform on par with Transformer models while being much more memory-efficient and faster on long sequences. |
| [UniLMv2](https://ritvik19.medium.com/papers-explained-unilmv2-5a044ca7c525) | February 2020 | Utilizes a pseudo-masked language model (PMLM) for both autoencoding and partially autoregressive language modeling tasks,significantly advancing the capabilities of language models in diverse NLP tasks. |
| [FastBERT](https://ritvik19.medium.com/papers-explained-37-fastbert-5bd246c1b432) | April 2020 | A speed-tunable encoder with adaptive inference time having branches at each transformer output to enable early outputs. |
| [MobileBERT](https://ritvik19.medium.com/papers-explained-36-mobilebert-933abbd5aaf1) | April 2020 | Compressed and faster version of the BERT, featuring bottleneck structures, optimized attention mechanisms, and knowledge transfer. |
| [Longformer](https://ritvik19.medium.com/papers-explained-38-longformer-9a08416c532e) | April 2020 | Introduces a linearly scalable attention mechanism, allowing handling texts of exteded length. |
| [GPT 3](https://ritvik19.medium.com/papers-explained-66-gpt-3-352f5a1b397) | May 2020 | Demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance. |
| [DeBERTa](https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d) | June 2020 | Enhances BERT and RoBERTa through disentangled attention mechanisms, an enhanced mask decoder, and virtual adversarial training. |
| [DeBERTa v2](https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d#f5e1) | June 2020 | Enhanced version of the DeBERTa featuring a new vocabulary, nGiE integration, optimized attention mechanisms, additional model sizes, and improved tokenization. |
| [T5 v1.1](https://ritvik19.medium.com/papers-explained-44-t5-9d974a3b7957#773b) | July 2020 | An enhanced version of the original T5 model, featuring improvements such as GEGLU activation, no dropout in pre-training, exclusive pre-training on C4, no parameter sharing between embedding and classifier layers. |
| [mT5](https://ritvik19.medium.com/papers-explained-113-mt5-c61e03bc9218) | October 2020 | A multilingual variant of T5 based on T5 v1.1, pre-trained on a new Common Crawl-based dataset covering 101 languages (mC4). |
| [Codex](https://ritvik19.medium.com/papers-explained-45-codex-caca940feb31) | July 2021 | A GPT language model finetuned on publicly available code from GitHub. |
| [FLAN](https://ritvik19.medium.com/papers-explained-46-flan-1c5e0d5db7c9) | September 2021 | An instruction-tuned language model developed through finetuning on various NLP datasets described by natural language instructions. |
| [T0](https://ritvik19.medium.com/papers-explained-74-t0-643a53079fe) | October 2021 | A fine tuned encoder-decoder model on a multitask mixture covering a wide variety of tasks, attaining strong zero-shot performance on several standard datasets. |
| [WebGPT](https://ritvik19.medium.com/papers-explained-123-webgpt-5bb0dd646b32) | December 2021 | A fine-tuned GPT-3 model utilizing text-based web browsing, trained via imitation learning and human feedback, enhancing its ability to answer long-form questions with factual accuracy. |
| [Gopher](https://ritvik19.medium.com/papers-explained-47-gopher-2e71bbef9e87) | December 2021 | Provides a comprehensive analysis of the performance of various Transformer models across different scales upto 280B on 152 tasks. |
| [LaMDA](https://ritvik19.medium.com/papers-explained-76-lamda-a580ebba1ca2) | January 2022 | Transformer based models specialized for dialog, which are pre-trained on public dialog data and web text. |
| [Instruct GPT](https://ritvik19.medium.com/papers-explained-48-instructgpt-e9bcd51f03ec) | March 2022 | Fine-tuned GPT using supervised learning (instruction tuning) and reinforcement learning from human feedback to align with user intent. |
| [CodeGen](https://ritvik19.medium.com/papers-explained-125-codegen-a6bae5c1f7b5) | March 2022 | An LLM trained for program synthesis using input-output examples and natural language descriptions. |
| [Chinchilla](https://ritvik19.medium.com/papers-explained-49-chinchilla-a7ad826d945e) | March 2022 | Investigated the optimal model size and number of tokens for training a transformer LLM within a given compute budget (Scaling Laws). |
| [PaLM](https://ritvik19.medium.com/papers-explained-50-palm-480e72fa3fd5) | April 2022 | A 540-B parameter, densely activated, Transformer, trained using Pathways, (ML system that enables highly efficient training across multiple TPU Pods). |
| [GPT-NeoX-20B](https://ritvik19.medium.com/papers-explained-78-gpt-neox-20b-fe39b6d5aa5b) | April 2022 | An autoregressive LLM trained on the Pile, and the largest dense model that had publicly available weights at the time of submission. |
| [OPT](https://ritvik19.medium.com/papers-explained-51-opt-dacd9406e2bd) | May 2022 | A suite of decoder-only pre-trained transformers with parameter ranges from 125M to 175B. OPT-175B being comparable to GPT-3. |
| [Flan T5, Flan PaLM](https://ritvik19.medium.com/papers-explained-75-flan-t5-flan-palm-caf168b6f76) | October 2022 | Explores instruction fine tuning with a particular focus on scaling the number of tasks, scaling the model size, and fine tuning on chain-of-thought data. |
| [BLOOM](https://ritvik19.medium.com/papers-explained-52-bloom-9654c56cd2) | November 2022 | A 176B-parameter open-access decoder-only transformer, collaboratively developed by hundreds of researchers, aiming to democratize LLM technology. |
| [BLOOMZ, mT0](https://ritvik19.medium.com/papers-explained-99-bloomz-mt0-8932577dcd1d) | November 2022 | Applies Multitask prompted fine tuning to the pretrained multilingual models on English tasks with English prompts to attain task generalization to non-English languages that appear only in the pretraining corpus. |
| [Galactica](https://ritvik19.medium.com/papers-explained-53-galactica-1308dbd318dc) | November 2022 | An LLM trained on scientific data thus specializing in scientific knowledge. |
| [ChatGPT](https://ritvik19.medium.com/papers-explained-54-chatgpt-78387333268f) | November 2022 | An interactive model designed to engage in conversations, built on top of GPT 3.5. |
| [Self Instruct](https://ritvik19.medium.com/papers-explained-112-self-instruct-5c192580103a) | December 2022 | A framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations. |
| [LLaMA](https://ritvik19.medium.com/papers-explained-55-llama-c4f302809d6b) | February 2023 | A collection of foundation LLMs by Meta ranging from 7B to 65B parameters, trained using publicly available datasets exclusively. |
| [Toolformer](https://ritvik19.medium.com/papers-explained-140-toolformer-d21d496b6812) | February 2023 | An LLM trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. |
| [Alpaca](https://ritvik19.medium.com/papers-explained-56-alpaca-933c4d9855e5) | March 2023 | A fine-tuned LLaMA 7B model, trained on instruction-following demonstrations generated in the style of self-instruct using text-davinci-003. |
| [GPT 4](https://ritvik19.medium.com/papers-explained-67-gpt-4-fc77069b613e) | March 2023 | A multimodal transformer model pre-trained to predict the next token in a document, which can accept image and text inputs and produce text outputs. |
| [Vicuna](https://ritvik19.medium.com/papers-explained-101-vicuna-daed99725c7e) | March 2023 | A 13B LLaMA chatbot fine tuned on user-shared conversations collected from ShareGPT, capable of generating more detailed and well-structured answers compared to Alpaca. |
| [BloombergGPT](https://ritvik19.medium.com/papers-explained-120-bloomberggpt-4bedd52ef54b) | March 2023 | A 50B language model train on general purpose and domain specific data to support a wide range of tasks within the financial industry. |
| [Pythia](https://ritvik19.medium.com/papers-explained-121-pythia-708284c32964) | April 2023 | A suite of 16 LLMs all trained on public data seen in the exact same order and ranging in size from 70M to 12B parameters. |
| [WizardLM](https://ritvik19.medium.com/papers-explained-127-wizardlm-65099705dfa3) | April 2023 | Introduces Evol-Instruct, a method to generate large amounts of instruction data with varying levels of complexity using LLM instead of humans to fine tune a Llama model  |
| [CodeGen2](https://ritvik19.medium.com/papers-explained-codegen2-d2690d7eb831) | May 2023 | Proposes an approach to make the training of LLMs for program synthesis more efficient by unifying key components of model architectures, learning methods, infill sampling, and data distributions |
| [PaLM 2](https://ritvik19.medium.com/papers-explained-58-palm-2-1a9a23f20d6c) | May 2023 | Successor of PALM, trained on a mixture of different pre-training objectives in order to understand different aspects of language. |
| [LIMA](https://ritvik19.medium.com/papers-explained-57-lima-f9401a5760c3) | May 2023 | A LLaMa model fine-tuned on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling. |
| [Gorilla](https://ritvik19.medium.com/papers-explained-139-gorilla-79f4730913e9) | May 2023 | A retrieve-aware finetuned LLaMA-7B model, specifically for API calls. |
| [Orca](https://ritvik19.medium.com/papers-explained-160-orca-928eff06e7f9) | June 2023 | Presents a novel approach that addresses the limitations of instruction tuning by leveraging richer imitation signals, scaling tasks and instructions, and utilizing a teacher assistant to help with progressive learning. |
| [Falcon](https://ritvik19.medium.com/papers-explained-59-falcon-26831087247f) | June 2023 | An Open Source LLM trained on properly filtered and deduplicated web data alone. |
| [Phi-1](https://ritvik19.medium.com/papers-explained-114-phi-1-14a8dcc77ce5) | June 2023 | An LLM for code, trained using a textbook quality data from the web and synthetically generated textbooks and exercises with GPT-3.5. |
| [WizardCoder](https://ritvik19.medium.com/papers-explained-wizardcoder-a12ecb5b93b6) | June 2023 | Enhances the performance of the open-source Code LLM, StarCoder, through the application of Code Evol-Instruct. |
| [LLaMA 2](https://ritvik19.medium.com/papers-explained-60-llama-v2-3e415c5b9b17) | July 2023 | Successor of LLaMA. LLaMA 2-Chat is optimized for dialogue use cases. |
| [Tool LLM](https://ritvik19.medium.com/papers-explained-141-tool-llm-856f99e79f55) | July 2023 | A LLaMA model finetuned on an instruction-tuning dataset for tool use, automatically created using ChatGPT. |
| [Humpback](https://ritvik19.medium.com/papers-explained-61-humpback-46992374fc34) | August 2023 | LLaMA finetuned using Instrustion backtranslation. |
| [Code LLaMA](https://ritvik19.medium.com/papers-explained-62-code-llama-ee266bfa495f) | August 2023 | LLaMA 2 based LLM for code. |
| [WizardMath](https://ritvik19.medium.com/papers-explained-129-wizardmath-265e6e784341) | August 2023 | Proposes Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method, applied to Llama-2 to enhance the mathematical reasoning abilities. |
| [LLaMA 2 Long](https://ritvik19.medium.com/papers-explained-63-llama-2-long-84d33c26d14a) | September 2023 | A series of long context LLMs s that support effective context windows of up to 32,768 tokens. |
| [Phi-1.5](https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a) | September 2023 | Follows the phi-1 approach, focusing this time on common sense reasoning in natural language. |
| [Mistral 7B](https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580) | October 2023 | Leverages grouped-query attention for faster inference, coupled with sliding window attention to effectively handle sequences of arbitrary length with a reduced inference cost. |
| [Llemma](https://ritvik19.medium.com/papers-explained-69-llemma-0a17287e890a) | October 2023 | An LLM for mathematics, formed by continued pretraining of Code Llama on a mixture of scientific papers, web data containing mathematics, and mathematical code. |
| [CodeFusion](https://ritvik19.medium.com/papers-explained-70-codefusion-fee6aba0149a) | October 2023 | A diffusion code generation model that iteratively refines entire programs based on encoded natural language, overcoming the limitation of auto-regressive models in code generation by allowing reconsideration of earlier tokens. |
| [Zephyr 7B](https://ritvik19.medium.com/papers-explained-71-zephyr-7ec068e2f20b) | October 2023 | Utilizes dDPO and AI Feedback (AIF) preference data to achieve superior intent alignment in chat-based language modeling. |
| [Orca 2](https://ritvik19.medium.com/papers-explained-161-orca-2-b6ffbccd1eef) | November 2023 | Introduces Cautious Reasoning for training smaller models to select the most effective solution strategy based on the problem at hand, by crafting data with task-specific system instruction(s) corresponding to the chosen strategy in order to obtain teacher responses for each task and replacing the student’s system instruction with a generic one vacated of details of how to approach the task. |
| [Phi-2](https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a) | December 2023 | A 2.7B model, developed to explore whether emergent abilities achieved by large-scale language models can also be achieved at a smaller scale using strategic choices for training, such as data selection. |
| [TinyLlama](https://ritvik19.medium.com/papers-explained-93-tinyllama-6ef140170da9) | January 2024 | A  1.1B language model built upon the architecture and tokenizer of Llama 2, pre-trained on around 1 trillion tokens for approximately 3 epochs, leveraging FlashAttention and Grouped Query Attention, to achieve better computational efficiency. |
| [Mixtral 8x7B](https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745) | January 2024 | A Sparse Mixture of Experts language model trained with multilingual data using a context size of 32k tokens. |
| [H2O Danube 1.8B](https://ritvik19.medium.com/papers-explained-111-h2o-danube-1-8b-b790c073d257) | January 2024 | A language model trained on 1T tokens following the core principles of LLama 2 and Mistral, leveraging and refining various techniques for pre-training large language models. |
| [OLMo](https://ritvik19.medium.com/papers-explained-98-olmo-fdc358326f9b) | February 2024 | A state-of-the-art, truly open language model and framework that includes training data, code, and tools for building, studying, and advancing language models. |
| [Orca Math](https://ritvik19.medium.com/papers-explained-163-orca-math-ae6a157ce48d) | February 2024 | A fine tuned Mistral-7B that excels at math problems without external tools, utilizing a high-quality synthetic dataset of 200K problems created through multi-agent collaboration and an iterative learning process that involves practicing problem-solving, receiving feedback, and learning from preference pairs incorporating the model's solutions and feedback. |
| [Gemma](https://ritvik19.medium.com/papers-explained-106-gemma-ca2b449321ac) | February 2024 | A family of 2B and 7B, state-of-the-art language models based on Google's Gemini models, offering advancements in language understanding, reasoning, and safety. |
| [Aya 101](https://ritvik19.medium.com/papers-explained-aya-101-d813ba17b83a) | Februray 2024 | A massively multilingual generative language model that follows instructions in 101 languages,trained by finetuning mT5. |
| [Hawk, Griffin](https://ritvik19.medium.com/papers-explained-131-hawk-griffin-dfc8c77f5dcd) |  February 2024 | Introduces Real Gated Linear Recurrent Unit Layer that forms the core of the new recurrent block, replacing Multi-Query Attention for better efficiency and scalability |
| [WRAP](https://ritvik19.medium.com/papers-explained-118-wrap-e563e009fe56) | March 2024 | Uses an off-the-shelf instruction-tuned model prompted to paraphrase documents on the web in specific styles to jointly pre-train LLMs on real and synthetic rephrases. |
| [Command R](https://ritvik19.medium.com/papers-explained-166-command-r-models-94ba068ebd2b) | March 2024 | An LLM optimized for retrieval-augmented generation and tool use, across multiple languages. |
| [DBRX](https://ritvik19.medium.com/papers-explained-119-dbrx-17c61739983c) | March 2024 | A 132B open, general-purpose fine grained Sparse MoE LLM surpassing GPT-3.5 and competitive with Gemini 1.0 Pro. |
| [Command R+](https://ritvik19.medium.com/papers-explained-166-command-r-models-94ba068ebd2b#c2b5) | April 2024 | Successor of Command R+ with improved performance for retrieval-augmented generation and tool use, across multiple languages. |
| [Mixtral 8x22B](https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745#20f3) | April 2024 | A open-weight AI model optimised for performance and efficiency, with capabilities such as fluency in multiple languages, strong mathematics and coding abilities, and precise information recall from large documents. |
| [CodeGemma](https://ritvik19.medium.com/papers-explained-124-codegemma-85faa98af20d) | April 2024 | Open code models based on Gemma models by further training on over 500 billion tokens of primarily code. |
| [RecurrentGemma](https://ritvik19.medium.com/papers-explained-132-recurrentgemma-52732d0f4273) | April 2024 | Based on Griffin, uses a combination of linear recurrences and local attention instead of global attention to model long sequences efficiently. |
| [Rho-1](https://ritvik19.medium.com/papers-explained-132-rho-1-788125e42241) | April 2024 | Introduces Selective Language Modelling that optimizes the loss only on tokens that align with a desired distribution, utilizing a reference model to score and select tokens. |
| [Phi-3](https://ritvik19.medium.com/papers-explained-130-phi-3-0dfc951dc404) | April 2024 | A series of language models trained on heavily filtered web and synthetic data set, achieving performance comparable to much larger models like Mixtral 8x7B and GPT-3.5. |
| [Open ELM](https://ritvik19.medium.com/papers-explained-133-open-elm-864f6b28a6ab) | April 2024 | A fully open language model designed to enhance accuracy while using fewer parameters and pre-training tokens. Utilizes a layer-wise scaling strategy to allocate smaller dimensions in early layers, expanding in later layers. |
| [H2O Danube2 1.8B](https://ritvik19.medium.com/papers-explained-111-h2o-danube-1-8b-b790c073d257#00d8) | April 2024 |  An updated version of the original H2O-Danube model, with improvements including removal of sliding window attention, changes to the tokenizer, and adjustments to the training data, resulting in significant performance enhancements. |
| [Granite Code Models](https://ritvik19.medium.com/paper-explained-144-granite-code-models-e1a92678739b) | May 2024 | A family of code models ranging from 3B to 34B trained on 3.5-4.5T tokens of code written in 116 programming languages. |
| [Codestral 22B](https://medium.com/dair-ai/papers-explained-mistral-7b-b9632dedf580#057b) | May 2024 | An open-weight model designed for code generation tasks, trained on over 80 programming languages, and licensed under the Mistral AI Non-Production License, allowing developers to use it for research and testing purposes. |
| [Aya 23](https://ritvik19.medium.com/papers-explained-151-aya-23-d01605c3ee80) | May 2024 | A family of multilingual language models supporting 23 languages, designed to balance breadth and depth by allocating more capacity to fewer languages during pre-training. |
| [Gemma 2](https://ritvik19.medium.com/papers-explained-157-gemma-2-f1b75b56b9f2) | June 2024 | Utilizes interleaving local-global attentions and group-query attention, trained with knowledge distillation instead of next token prediction to achieve competitive performance comparable with larger models. |
| [Orca 3 (Agent Instruct)](https://ritvik19.medium.com/papers-explained-164-orca-3-agent-instruct-41340505af36) | June 2024 | A fine tuned Mistral-7B through Generative Teaching via synthetic data generated through the proposed AgentInstruct framework, which generates both the prompts and responses, using only raw data sources like text documents and code files as seeds. |
| [Mathstral](https://medium.com/dair-ai/papers-explained-mistral-7b-b9632dedf580#0fbe) | July 2024 | a 7B model designed for math reasoning and scientific discovery based on Mistral 7B specializing in STEM subjects. |
| [Mistral Nemo](https://medium.com/dair-ai/papers-explained-mistral-7b-b9632dedf580#37cd) | July 2024 | A 12B Language Model built in collaboration between Mistral and NVIDIA, featuring a context window of 128K, an efficient tokenizer and trained with quantization awareness, enabling FP8 inference without any performance loss. |
| [Mistral Large 2](https://medium.com/dair-ai/papers-explained-mistral-7b-b9632dedf580#301d) | July 2024 | A 123B model, offers significant improvements in code generation, mathematics, and reasoning capabilities, advanced function calling, a 128k context window, and supports dozens of languages and over 80 coding languages. |


## Multi Modal Language Models

| Paper | Date | Description |
|---|---|---|
| [BLIP](https://ritvik19.medium.com/papers-explained-154-blip-6d85c80a744d) | February 2022 | A Vision-Language Pre-training (VLP) framework that introduces Multimodal mixture of Encoder-Decoder (MED) and Captioning and Filtering (CapFilt), a new dataset bootstrapping method for learning from noisy image-text pairs. |
| [Flamingo](https://ritvik19.medium.com/papers-explained-82-flamingo-8c124c394cdb) | April 2022 | Visual Language Models enabling seamless handling of interleaved visual and textual data, and facilitating few-shot learning on large-scale web corpora. |
| [BLIP 2](https://ritvik19.medium.com/papers-explained-155-blip-2-135fff70bf65) | January 2023 | A Vision-Language Pre-training (VLP) framework that proposes Q-Former, a trainable module to bridge the gap between a frozen image encoder and a frozen LLM to bootstrap vision-language pre-training. |
| [LLaVA 1](https://ritvik19.medium.com/papers-explained-102-llava-1-eb0a3db7e43c) | April 2023 | A large multimodal model connecting CLIP and Vicuna trained end-to-end on instruction-following data generated through GPT-4 from image-text pairs. |
| [InstructBLIP](https://ritvik19.medium.com/papers-explained-156-instructblip-c3cf3291a823) | May 2023 | Introduces instruction-aware Query Transformer to extract informative features tailored to the given instruction to study vision-language instruction tuning based on the pretrained BLIP-2 models. |
| [GPT-4V](https://ritvik19.medium.com/papers-explained-68-gpt-4v-6e27c8a1d6ea) | September 2023 | A multimodal model that combines text and vision capabilities, allowing users to instruct it to analyze image inputs. |
| [LLaVA 1.5](https://ritvik19.medium.com/papers-explained-103-llava-1-5-ddcb2e7f95b4) | October 2023 | An enhanced version of the LLaVA model that incorporates a CLIP-ViT-L-336px with an MLP projection and academic-task-oriented VQA data to set new benchmarks in large multimodal models (LMM) research. |
| [Gemini 1.0](https://ritvik19.medium.com/papers-explained-80-gemini-1-0-97308ef96fcd) | December 2023 | A family of highly capable multi-modal models, trained jointly across image, audio, video, and text data for the purpose of building a model with strong generalist capabilities across modalities. |
| [MoE-LLaVA](https://ritvik19.medium.com/papers-explained-104-moe-llava-cf14fda01e6f) | January 2024 | A MoE-based sparse LVLM framework that activates only the top-k experts through routers during deployment, maintaining computational efficiency while achieving comparable performance to larger models. |
| [LLaVA 1.6](https://ritvik19.medium.com/papers-explained-107-llava-1-6-a312efd496c5) | January 2024 | An improved version of a LLaVA 1.5 with enhanced reasoning, OCR, and world knowledge capabilities, featuring increased image resolution |
| [Gemini 1.5 Pro](https://ritvik19.medium.com/papers-explained-105-gemini-1-5-pro-029bbce3b067) | February 2024 | A highly compute-efficient multimodal mixture-of-experts model that excels in long-context retrieval tasks and understanding across text, video, and audio modalities. |
| [MM1](https://ritvik19.medium.com/papers-explained-117-mm1-c579142bcdc0) | March 2024 | A multimodal llm that combines a ViT-H image encoder with 378x378px resolution, pretrained on a data mix of image-text documents and text-only documents, scaled up to 3B, 7B, and 30B parameters for enhanced performance across various tasks |
| [Phi 3 Vision](https://ritvik19.medium.com/papers-explained-130-phi-3-0dfc951dc404#7ba6) | May 2024 | First multimodal model in the Phi family, bringing the ability to reason over images and extract and reason over text from images. |
| [GPT-4o](https://ritvik19.medium.com/papers-explained-67-gpt-4-fc77069b613e#49a8) | May 2024 | An omni model accepting and generating various types of inputs and outputs, including text, audio, images, and video. |
| [Gemini 1.5 Flash](https://ritvik19.medium.com/papers-explained-142-gemini-1-5-flash-415e2dc6a989) | May 2024 | A more lightweight variant of the Gemini 1.5 pro, designed for efficiency with minimal regression in quality, making it suitable for applications where compute resources are limited. |
| [Chameleon](https://ritvik19.medium.com/papers-explained-143-chameleon-6cddfdbceaa8) | May 2024 | A family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. |
| [GPT-4o mini](https://ritvik19.medium.com/papers-explained-67-gpt-4-fc77069b613e#17c9) | July 2024 | A cost-efficient small model that outperforms GPT-4 on chat preferences, enabling a broad range of tasks with low latency and supporting text, vision, and multimodal inputs and outputs. |


## Retrieval and Representation Learning

| Paper | Date | Description |
|---|---|---|
| [Dense Passage Retriever](https://ritvik19.medium.com/papers-explained-86-dense-passage-retriever-c4742fdf27ed) | April 2020 | Shows that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual encoder framework. |
| [ColBERT](https://medium.com/@ritvik19/papers-explained-88-colbert-fe2fd0509649) | April 2020 | Introduces a late interaction architecture that adapts deep LMs (in particular, BERT) for efficient retrieval. |
| [CLIP](https://ritvik19.medium.com/papers-explained-100-clip-f9873c65134) | February 2021 | A vision system that learns image representations from raw text-image pairs through pre-training, enabling zero-shot transfer to various downstream tasks. |
| [ColBERTv2](https://ritvik19.medium.com/papers-explained-89-colbertv2-7d921ee6e0d9) | December 2021 | Couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction. |
| [Matryoshka Representation Learning](https://ritvik19.medium.com/papers-explained-matryoshka-representation-learning-e7a139f6ad27) | May 2022 | Encodes information at different granularities and allows a flexible representation that can adapt to multiple downstream tasks with varying computational resources using a single embedding. |
| [E5](https://ritvik19.medium.com/papers-explained-90-e5-75ea1519efad) | December 2022 | A family of text embeddings trained in a contrastive manner with weak supervision signals from a curated large-scale text pair dataset CCPairs. |
| [SigLip](https://ritvik19.medium.com/papers-explained-152-siglip-011c48f9d448) | March 2023 | A simple pairwise Sigmoid loss function for Language-Image Pre-training that operates solely on image-text pairs, allowing for larger batch sizes and better performance at smaller batch sizes. |
| [E5 Mistral 7B](https://ritvik19.medium.com/papers-explained-91-e5-mistral-7b-23890f40f83a) | December 2023 | Leverages proprietary LLMs to generate diverse synthetic data to fine tune open-source decoder-only LLMs for hundreds of thousands of text embedding tasks. |
| [Nomic Embed Text v1](https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2) | February 2024 | A 137M parameter, open-source English text embedding model with an 8192 context length that outperforms OpenAI's models on both short and long-context tasks. |
| [Nomic Embed Text v1.5](https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2#2119) | February 2024 | An advanced text embedding model that utilizes Matryoshka Representation Learning to offer flexible embedding sizes with minimal performance trade-offs |
| [NV Embed](https://ritvik19.medium.com/papers-explained-168-nv-embed-48bd25d83258) | May 2024 | Introduces architectural innovations and training recipe to significantly enhance LLMs performance in general-purpose text embedding tasks. |
| [Nomic Embed Vision v1 and v1.5](https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2#486b) | June 2024 | Aligns a Vision Encoder with the existing text encoders without destroying the downstream performance of the text encoder, to attain a unified multimodal latent space. |


## Parameter Efficient Fine Tuning

| Paper | Date | Description |
|---|---|---|
| [LoRA](https://ritvik19.medium.com/papers-explained-lora-a48359cecbfa) | July 2021 | Introduces trainable rank decomposition matrices into each layer of a pre-trained Transformer model, significantly reducing the number of trainable parameters for downstream tasks. |
| [QLoRA](https://ritvik19.medium.com/papers-explained-146-qlora-a6e7273bc630) | May 2023 | Allows efficient training of large models on limited GPU memory, through innovations like 4-bit NormalFloat (NF4), double quantization and paged optimisers. |
| [LongLoRA](https://ritvik19.medium.com/papers-explained-147-longlora-24f095b93611) | September 2023 | Enables context extension for large language models, achieving significant computation savings through sparse local attention and parameter-efficient fine-tuning. |

## Compression, Pruning, Quantization

| Paper | Date | Description |
|---|---|---|
| [LLMLingua](https://ritvik19.medium.com/papers-explained-136-llmlingua-f9b2f53f5f9b) | October 2023 | A novel coarse-to-fine prompt compression method, incorporating a budget controller, an iterative token-level compression algorithm, and distribution alignment, achieving up to 20x compression with minimal performance loss. |
| [LongLLMLingua](https://ritvik19.medium.com/papers-explained-137-longllmlingua-45961fa703dd) | October 2023 | A novel approach for prompt compression to enhance performance in long context scenarios using question-aware compression and document reordering. |
| [LLMLingua2](https://ritvik19.medium.com/papers-explained-138-llmlingua-2-510c752368a8) | March 2024 | A novel approach to task-agnostic prompt compression, aiming to enhance generalizability, using  data distillation and leveraging a Transformer encoder for token classification. |

## Vision Models

| Paper | Date | Description |
|---|---|---|
| [Vision Transformer](https://ritvik19.medium.com/papers-explained-25-vision-transformers-e286ee8bc06b) | October 2020 | Images are segmented into patches, which are treated as tokens and a sequence of linear embeddings of these patches are input to a Transformer |
| [DeiT](https://ritvik19.medium.com/papers-explained-39-deit-3d78dd98c8ec) | December 2020 | A convolution-free vision transformer that uses a teacher-student strategy with attention-based distillation tokens. |
| [Swin Transformer](https://ritvik19.medium.com/papers-explained-26-swin-transformer-39cf88b00e3e) | March 2021 | A hierarchical vision transformer that uses shifted windows to addresses the challenges of adapting the transformer model to computer vision. |
| [BEiT](https://ritvik19.medium.com/papers-explained-27-beit-b8c225496c01) | June 2021 | Utilizes a masked image modeling task inspired by BERT in, involving image patches and visual tokens to pretrain vision Transformers. |
| [MobileViT](https://ritvik19.medium.com/papers-explained-40-mobilevit-4793f149c434) | October 2021 | A lightweight vision transformer designed for mobile devices, effectively combining the strengths of CNNs and ViTs. |
| [Masked AutoEncoder](https://ritvik19.medium.com/papers-explained-28-masked-autoencoder-38cb0dbed4af) | November 2021 | An encoder-decoder architecture that reconstructs input images by masking random patches and leveraging a high proportion of masking for self-supervision. |

## Convolutional Neural Networks

| Paper | Date | Description |
|---|---|---|
| [Lenet](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4f26) | December 1998 | Introduced Convolutions. |
| [Alex Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f7c6) | September 2012 | Introduced ReLU activation and Dropout to CNNs. Winner ILSVRC 2012. |
| [VGG](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#c122) | September 2014 | Used large number of filters of small size in each layer to learn complex features. Achieved SOTA in ILSVRC 2014. |
| [Inception Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3) | September 2014 | Introduced Inception Modules consisting of multiple parallel convolutional layers, designed to recognize different features at multiple scales. |
| [Inception Net v2 / Inception Net v3](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3) | December 2015 | Design Optimizations of the Inception Modules which improved performance and accuracy. |
| [Res Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f761) | December 2015 | Introduced residual connections, which are shortcuts that bypass one or more layers in the network. Winner ILSVRC 2015. |
| [Inception Net v4 / Inception ResNet](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#83ad) | February 2016 | Hybrid approach combining Inception Net and ResNet. |
| [Dense Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#65e8) | August 2016 | Each layer receives input from all the previous layers, creating a dense network of connections between the layers, allowing to learn more diverse features. |
| [Xception](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#bc70) | October 2016 | Based on InceptionV3 but uses depthwise separable convolutions instead on inception modules. |
| [Res Next](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#90bd) | November 2016 | Built over ResNet, introduces the concept of grouped convolutions, where the filters in a convolutional layer are divided into multiple groups. |
| [Mobile Net V1](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#3cb5) | April 2017 | Uses depthwise separable convolutions to reduce the number of parameters and computation required. |
| [Mobile Net V2](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4440) | January 2018 | Built upon the MobileNetv1 architecture, uses inverted residuals and linear bottlenecks. |
| [Mobile Net V3](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#8eb6) | May 2019 | Uses AutoML to find the best possible neural network architecture for a given problem. |
| [Efficient Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#560a) | May 2019 | Uses a compound scaling method to scale the network's depth, width, and resolution to achieve a high accuracy with a relatively low computational cost. |
| [NF Net](https://ritvik19.medium.com/papers-explained-84-nf-net-b8efa03d6b26) | February 2021 | An improved class of Normalizer-Free ResNets that implement batch-normalized networks, offer faster training times, and introduce an adaptive gradient clipping technique to overcome instabilities associated with deep ResNets. |
| [Conv Mixer](https://ritvik19.medium.com/papers-explained-29-convmixer-f073f0356526) | January 2022 | Processes image patches using standard convolutions for mixing spatial and channel dimensions. |
| [ConvNeXt](https://ritvik19.medium.com/papers-explained-92-convnext-d13385d9177d) | January 2022 | A pure ConvNet model, evolved from standard ResNet design, that competes well with Transformers in accuracy and scalability. |
| [ConvNeXt V2](https://ritvik19.medium.com/papers-explained-94-convnext-v2-2ecdabf2081c) | January 2023 | Incorporates a fully convolutional MAE framework and a Global Response Normalization (GRN) layer, boosting performance across multiple benchmarks. |

## Object Detection

| Paper | Date | Description |
|---|---|---|
| [SSD](https://ritvik19.medium.com/papers-explained-31-single-shot-multibox-detector-14b0aa2f5a97) | December 2015 | Discretizes bounding box outputs over a span of various scales and aspect ratios per feature map. |
| [Feature Pyramid Network](https://ritvik19.medium.com/papers-explained-21-feature-pyramid-network-6baebcb7e4b8) | December 2016 | Leverages the inherent multi-scale hierarchy of deep convolutional networks to efficiently construct feature pyramids. |
| [Focal Loss](https://ritvik19.medium.com/papers-explained-22-focal-loss-for-dense-object-detection-retinanet-733b70ce0cb1) | August 2017 | Addresses class imbalance in dense object detectors by down-weighting the loss assigned to well-classified examples. |
| [DETR](https://ritvik19.medium.com/papers-explained-79-detr-bcdd53355d9f) | May 2020 | A novel object detection model that treats object detection as a set prediction problem, eliminating the need for hand-designed components. |

## Region-based Convolutional Neural Networks

| Paper | Date | Description |
|---|---|---|
| [RCNN](https://ritvik19.medium.com/papers-explained-14-rcnn-ede4db2de0ab) | November 2013 | Uses selective search for region proposals, CNNs for feature extraction, SVM for classification followed by box offset regression. |
| [Fast RCNN](https://ritvik19.medium.com/papers-explained-15-fast-rcnn-28c1792dcee0) | April 2015 | Processes entire image through CNN, employs RoI Pooling to extract feature vectors from ROIs, followed by classification and BBox regression. |
| [Faster RCNN](https://ritvik19.medium.com/papers-explained-16-faster-rcnn-a7b874ffacd9) | June 2015 | A region proposal network (RPN) and a Fast R-CNN detector, collaboratively predict object regions by sharing convolutional features. |
| [Mask RCNN](https://ritvik19.medium.com/papers-explained-17-mask-rcnn-82c64bea5261) | March 2017 | Extends Faster R-CNN to solve instance segmentation tasks, by adding a branch for predicting an object mask in parallel with the existing branch. |
| [Cascade RCNN](https://ritvik19.medium.com/papers-explained-77-cascade-rcnn-720b161d86e4) | December 2017 | Proposes a multi-stage approach where detectors are trained with progressively higher IoU thresholds, improving selectivity against false positives. |

## Document AI

| Paper | Date | Description |
|---|---|---|
| [Table Net](https://ritvik19.medium.com/papers-explained-18-tablenet-3d4c62269bb3) | January 2020 | An end-to-end deep learning model designed for both table detection and structure recognition. |
| [Donut](https://ritvik19.medium.com/papers-explained-20-donut-cb1523bf3281) | November 2021 | An OCR-free Encoder-Decoder Transformer model. The encoder takes in images, decoder takes in prompts & encoded images to generate the required text. |
| [DiT](https://ritvik19.medium.com/papers-explained-19-dit-b6d6eccd8c4e) | March 2022 | An Image Transformer pre-trained (self-supervised) on document images |
| [UDoP](https://ritvik19.medium.com/papers-explained-42-udop-719732358ab4) | December 2022 | Integrates text, image, and layout information through a Vision-Text-Layout Transformer, enabling unified representation. |
| [DocLLM](https://ritvik19.medium.com/papers-explained-87-docllm-93c188edfaef) | January 2024 | A lightweight extension to traditional LLMs that focuses on reasoning over visual documents, by incorporating textual semantics and spatial layout without expensive image encoders. |

## Layout Transformers

| Paper | Date | Description |
|---|---|---|
| [Layout LM](https://ritvik19.medium.com/papers-explained-10-layout-lm-32ec4bad6406) | December 2019 | Utilises BERT as the backbone, adds two new input embeddings: 2-D position embedding and image embedding (Only for downstream tasks). |
| [LamBERT](https://ritvik19.medium.com/papers-explained-41-lambert-8f52d28f20d9) | February 2020 | Utilises RoBERTa as the backbone and adds Layout embeddings along with relative bias. |
| [Layout LM v2](https://ritvik19.medium.com/papers-explained-11-layout-lm-v2-9531a983e659) | December 2020 | Uses a multi-modal Transformer model, to integrate text, layout, and image in the pre-training stage, to learn end-to-end cross-modal interaction. |
| [Structural LM](https://ritvik19.medium.com/papers-explained-23-structural-lm-36e9df91e7c1) | May 2021 | Utilises BERT as the backbone and feeds text, 1D and (2D cell level) embeddings to the transformer model. |
| [Doc Former](https://ritvik19.medium.com/papers-explained-30-docformer-228ce27182a0) | June 2021 | Encoder-only transformer with a CNN backbone for visual feature extraction, combines text, vision, and spatial features through a multi-modal self-attention layer. |
| [LiLT](https://ritvik19.medium.com/papers-explained-12-lilt-701057ec6d9e) | February 2022 | Introduced Bi-directional attention complementation mechanism (BiACM) to accomplish the cross-modal interaction of text and layout. |
| [Layout LM V3](https://ritvik19.medium.com/papers-explained-13-layout-lm-v3-3b54910173aa) | April 2022 | A unified text-image multimodal Transformer to learn cross-modal representations, that imputs concatenation of text embedding and image embedding. |
| [ERNIE Layout](https://ritvik19.medium.com/papers-explained-24-ernie-layout-47a5a38e321b) | October 2022 | Reorganizes tokens using layout information, combines text and visual embeddings, utilizes multi-modal transformers with spatial aware disentangled attention. |

## Generative Adversarial Networks

| Paper | Date | Description |
|---|---|---|
| [Generative Adversarial Networks](https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#7041) | June 2014 | Introduces a framework where, a generative and a discriminative model, are trained simultaneously in a minimax game. |
| [Conditional Generative Adversarial Networks](https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#86aa) | November 2014 | A method for training GANs, enabling the generation based on specific conditions, by feeding them to both the generator and discriminator networks. |
| [Deep Convolutional Generative Adversarial Networks](https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#fe42) | November 2015 | Demonstrates the ability of CNNs for unsupervised learning using specific architectural constraints designed. |
| [Improved GAN](https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#9a55) | June 2016 | Presents a variety of new architectural features and training procedures that can be applied to the generative adversarial networks (GANs) framework. |
| [Wasserstein Generative Adversarial Networks](https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#6f8f) | January 2017 | An alternative GAN training algorithm that enhances learning stability, mitigates issues like mode collapse. |
| [Cycle GAN](https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#7f8b) | March 2017 | An approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples by leveraging adversarial losses and cycle consistency constraints, using two GANs. |


## Tabular Deep Learning

| Paper | Date | Description |
|---|---|---|
| [Entity Embeddings](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#932e) | April 2016 | Maps categorical variables into continuous vector spaces through neural network learning, revealing intrinsic properties. |
| [Wide and Deep Learning](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#bfdc) | June 2016 | Combines memorization of specific patterns with generalization of similarities. |
| [Deep and Cross Network](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#0017) | August 2017 | Combines the  a novel cross network with deep neural networks (DNNs) to efficiently learn feature interactions without manual feature engineering. |
| [Tab Transformer](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#48c4) | December 2020 | Employs multi-head attention-based Transformer layers to convert categorical feature embeddings into robust contextual embeddings. |
| [Tabular ResNet](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#46af) | June 2021 | An MLP with skip connections. |
| [Feature Tokenizer Transformer](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#1ab8) | June 2021 | Transforms all features (categorical and numerical) to embeddings and applies a stack of Transformer layers to the embeddings. |

## Datasets

| Paper | Date | Description |
|---|---|---|
| [Dolma](https://ritvik19.medium.com/papers-explained-97-dolma-a656169269cb) | January 2024 | An open corpus of three trillion tokens designed to support language model pretraining research. |
| [Aya Dataset](https://ritvik19.medium.com/papers-explained-108-aya-dataset-9e299ac74a19) | Februray 2024 | A human-curated instruction-following dataset that spans 65 languages, created to bridge the language gap in datasets for natural language processing. |

## Miscellaneous

| Paper | Date | Description |
|---|---|---|
| [ColD Fusion](https://ritvik19.medium.com/papers-explained-32-cold-fusion-452f33101a91) | December 2022 | A method enabling the benefits of multitask learning through distributed computation without data sharing and improving model performance. |
| [Are Emergent Abilities of Large Language Models a Mirage?](https://ritvik19.medium.com/papers-explained-are-emergent-abilities-of-large-language-models-a-mirage-4160cf0e44cb) | April 2023 | This paper presents an alternative explanation for emergent abilities, i.e. emergent abilities are created by the researcher’s choice of metrics, not fundamental changes in model family behaviour on specific tasks with scale. |
| [Scaling Data-Constrained Language Models](https://ritvik19.medium.com/papers-explained-85-scaling-data-constrained-language-models-2a4c18bcc7d3) | May 2023 | This study investigates scaling language models in data-constrained regimes. |
| [An In-depth Look at Gemini's Language Abilities](https://ritvik19.medium.com/papers-explained-81-an-in-depth-look-at-geminis-language-abilities-540ca9046d8e) | December 2023 | A third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results. |
| [DSPy](https://ritvik19.medium.com/papers-explained-135-dspy-fe8af7e35091) | October 2023 | A programming model that abstracts LM pipelines as text transformation graphs, i.e. imperative computation graphs where LMs are invoked through declarative modules, optimizing their use through a structured framework of signatures, modules, and teleprompters to automate and enhance text transformation tasks. |
| [Direct Preference Optimization](https://ritvik19.medium.com/papers-explained-148-direct-preference-optimization-d3e031a41be1) | December 2023 | A stable, performant, and computationally lightweight algorithm that fine-tunes llms to align with human preferences without the need for reinforcement learning, by directly optimizing for the policy best satisfying the preferences with a simple classification objective. |
| [RLHF Workflow](https://ritvik19.medium.com/papers-explained-149-rlhf-workflow-56b4e00019ed) | May 2024 | Provides a detailed recipe for  online iterative RLHF and achieves state-of-the-art performance on various benchmarks using fully open-source datasets. |
| [Monte Carlo Tree Self-refine](https://ritvik19.medium.com/papers-explained-167-monte-carlo-tree-self-refine-79bffb070c1a) | June 2024 | Integrates LLMs with Monte Carlo Tree Search to enhance performance in complex mathematical reasoning tasks, leveraging systematic exploration and heuristic self-refine mechanisms to improve decision-making frameworks. |

---

## Literature Reviewed
- [Convolutional Neural Networks](https://medium.com/dair-ai/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3)
- [Layout Transformers](https://medium.com/dair-ai/papers-explained-review-02-layout-transformers-b2d165c94ad5)
- [Region-based Convolutional Neural Networks](https://medium.com/dair-ai/papers-explained-review-03-rcnns-42c0a3974493)
- [Tabular Deep Learning](https://medium.com/dair-ai/papers-explained-review-04-tabular-deep-learning-776db04f965b)
- [Generative Adversarial Networks](https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e)

## Reading Lists
- [Language Models](https://ritvik19.medium.com/list/language-models-11b008ddc292)
- [Encoder Only Language Transformers](https://ritvik19.medium.com/list/encoderonly-language-transformers-0f2ff06e0309)
- [Decoder Only Language Transformers](https://ritvik19.medium.com/list/decoderonly-language-transformers-5448110c6046)
- [Language Models for Retrieval](https://ritvik19.medium.com/list/language-models-for-retrieval-3b6e14887105)
- [Small LLMs](https://ritvik19.medium.com/list/small-llms-41124d5c7c80)
- [LLMs for Code](https://ritvik19.medium.com/list/llms-for-code-e5360a1b353a)
- [GPT Models](https://ritvik19.medium.com/list/gpt-models-fa2cc801d840)
- [LLaMA Models](https://ritvik19.medium.com/list/llama-models-5b8ea07308cb)
- [Gemini / Gemma Models](https://ritvik19.medium.com/list/gemini-gemma-models-4cb7dfc50d42)
- [Wizard Models](https://ritvik19.medium.com/list/wizard-models-9b972e860683)
- [Orca Series](https://ritvik19.medium.com/list/orca-series-1c87367458fe)
- [LLM Lingua Series](https://ritvik19.medium.com/list/llm-lingua-series-2f61b47d0343)
- [Multi Task Language Models](https://ritvik19.medium.com/list/multi-task-language-models-e6a2a1e517e6)
- [Layout Aware Transformers](https://ritvik19.medium.com/list/layout-transformers-1ce4f291a9f0)
- [Retrieval and Representation Learning](https://ritvik19.medium.com/list/retrieval-and-representation-learning-bcd23de0bd8e)
- [Vision Transformers](https://ritvik19.medium.com/list/vision-transformers-61e6836230f1)
- [Multi Modal Transformers](https://ritvik19.medium.com/list/multi-modal-transformers-67453f215ecf)
- [Convolutional Neural Networks](https://ritvik19.medium.com/list/convolutional-neural-networks-5b875ce3b689)
- [Object Detection](https://ritvik19.medium.com/list/object-detection-bd9e6e21ca3e)
- [Region Based Convolutional Neural Networks](https://ritvik19.medium.com/list/rcnns-b51467f53dc9)
- [Document Information Processing](https://ritvik19.medium.com/list/document-information-processing-3cd900a34972)

---
Reach out to [Ritvik](https://twitter.com/RitvikRastogi19) or [Elvis](https://twitter.com/omarsar0) if you have any questions.

If you are interested to contribute, feel free to open a PR.

[Join our Discord](https://discord.gg/SKgkVT8BGJ)





