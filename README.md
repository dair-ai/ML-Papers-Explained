# ML Papers Explained

Explanations to key concepts in ML

## Language Models

| Paper | Date | Description |
|---|---|---|
| [Transformer](https://ritvik19.medium.com/papers-explained-01-transformer-474bb60a33f7) | June 2017 | An Encoder Decoder model, that introduced multihead attention mechanism for language translation task. |
| [Elmo](https://ritvik19.medium.com/papers-explained-33-elmo-76362a43e4) | February 2018 | Deep contextualized word representations that captures both intricate aspects of word usage and contextual variations across language contexts. |
| [GPT](https://ritvik19.medium.com/papers-explained-43-gpt-30b6f1e6d226) | June 2018 | A Decoder only transformer which is autoregressively pretrained and then finetuned for specific downstream tasks using task-aware input transformations. |
| [BERT](https://ritvik19.medium.com/papers-explained-02-bert-31e59abc0615) | October 2018 | Introduced pre-training for Encoder Transformers. Uses unified architecture across different tasks. |
| [Transformer XL](https://ritvik19.medium.com/papers-explained-34-transformerxl-2e407e780e8) | January 2019 | Extends the original Transformer model to handle longer sequences of text by introducing recurrence into the self-attention mechanism. |
| [GPT 2](https://ritvik19.medium.com/papers-explained-65-gpt-2-98d0a642e520) | February 2019 | Demonstrates that language models begin to learn various language processing tasks without any explicit supervision. | 
| [XLNet](https://ritvik19.medium.com/papers-explained-35-xlnet-ea0c3af96d49) | June 2019 | Extension of the Transformer-XL, pre-trained using a new method that combines ideas from AR and AE objectives. |
| [RoBERTa](https://ritvik19.medium.com/papers-explained-03-roberta-81db014e35b9) | July 2019 | Built upon BERT, by carefully optimizing hyperparameters and training data size to improve performance on various language tasks . |
| [Sentence BERT](https://ritvik19.medium.com/papers-explained-04-sentence-bert-5159b8e07f21) | August 2019 | A modification of BERT that uses siamese and triplet network structures to derive sentence embeddings that can be compared using cosine-similarity. |
| [Tiny BERT](https://ritvik19.medium.com/papers-explained-05-tiny-bert-5e36fe0ee173) | September 2019 | Uses attention transfer, and task specific distillation for distilling BERT. |
| [ALBERT](https://ritvik19.medium.com/papers-explained-07-albert-46a2a0563693) | September 2019 | Presents certain parameter reduction techniques to lower memory consumption and increase the training speed of BERT. |
| [Distil BERT](https://ritvik19.medium.com/papers-explained-06-distil-bert-6f138849f871) | October 2019 | Distills BERT on very large batches leveraging gradient accumulation, using dynamic masking and without the next sentence prediction objective. |
| [T5](https://ritvik19.medium.com/papers-explained-44-t5-9d974a3b7957) | October 2019 | A unified encoder-decoder framework that converts all text-based language problems into a text-to-text format. |
| [BART](https://ritvik19.medium.com/papers-explained-09-bart-7f56138175bd) | October 2019 | A Decoder pretrained to reconstruct the original text from corrupted versions of it. |
| [FastBERT](https://ritvik19.medium.com/papers-explained-37-fastbert-5bd246c1b432) | April 2020 | A speed-tunable encoder with adaptive inference time having branches at each transformer output to enable early outputs. |
| [MobileBERT](https://ritvik19.medium.com/papers-explained-36-mobilebert-933abbd5aaf1) | April 2020 | Compressed and faster version of the BERT, featuring bottleneck structures, optimized attention mechanisms, and knowledge transfer. |
| [Longformer](https://ritvik19.medium.com/papers-explained-38-longformer-9a08416c532e) | April 2020 | Introduces a linearly scalable attention mechanism, allowing handling texts of exteded length. |
| [GPT 3](https://ritvik19.medium.com/papers-explained-66-gpt-3-352f5a1b397) | May 2020 | Demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance. |
| [DeBERTa](https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d) | June 2020 | Enhances BERT and RoBERTa through disentangled attention mechanisms, an enhanced mask decoder, and virtual adversarial training. |
| [Codex](https://ritvik19.medium.com/papers-explained-45-codex-caca940feb31) | July 2021 | A GPT language model finetuned on publicly available code from GitHub. |
| [FLAN](https://ritvik19.medium.com/papers-explained-46-flan-1c5e0d5db7c9) | September 2021 | An instruction-tuned language model developed through finetuning on various NLP datasets described by natural language instructions. |
| [Gopher](https://ritvik19.medium.com/papers-explained-47-gopher-2e71bbef9e87) | December 2021 | Provides a comprehensive analysis of the performance of various Transformer models across different scales upto 280B on 152 tasks. |
| [Instruct GPT](https://ritvik19.medium.com/papers-explained-48-instructgpt-e9bcd51f03ec) | March 2022 | Fine-tuned GPT using supervised learning (instruction tuning) and reinforcement learning from human feedback to align with user intent. |
| [Chinchilla](https://ritvik19.medium.com/papers-explained-49-chinchilla-a7ad826d945e) | March 2022 | Investigated the optimal model size and number of tokens for training a transformer LLM within a given compute budget (Scaling Laws). |
| [PaLM](https://ritvik19.medium.com/papers-explained-50-palm-480e72fa3fd5) | April 2022 | A 540-B parameter, densely activated, Transformer, trained using Pathways, (ML system that enables highly efficient training across multiple TPU Pods). |
| [OPT](https://ritvik19.medium.com/papers-explained-51-opt-dacd9406e2bd) | May 2022 | A suite of decoder-only pre-trained transformers with parameter ranges from 125M to 175B. OPT-175B being comparable to GPT-3. |
| [BLOOM](https://ritvik19.medium.com/papers-explained-52-bloom-9654c56cd2) | November 2022 | A 176B-parameter open-access decoder-only transformer, collaboratively developed by hundreds of researchers, aiming to democratize LLM technology. |
| [Galactica](https://ritvik19.medium.com/papers-explained-53-galactica-1308dbd318dc) | November 2022 | An LLM trained on scientific data thus specializing in scientific knowledge. |
| [ChatGPT](https://ritvik19.medium.com/papers-explained-54-chatgpt-78387333268f) | November 2022 | An interactive model designed to engage in conversations, built on top of GPT 3.5. |
| [LLaMA](https://ritvik19.medium.com/papers-explained-55-llama-c4f302809d6b) | February 2023 | A collection of foundation LLMs by Meta ranging from 7B to 65B parameters, trained using publicly available datasets exclusively. |
| [Alpaca](https://ritvik19.medium.com/papers-explained-56-alpaca-933c4d9855e5) | March 2023 | A fine-tuned LLaMA 7B model, trained on instruction-following demonstrations generated in the style of self-instruct using text-davinci-003. |
| [GPT 4](https://ritvik19.medium.com/papers-explained-67-gpt-4-fc77069b613e) | March 2023 | A multimodal transformer model pre-trained to predict the next token in a document, which can accept image and text inputs and produce text outputs. |
| [PaLM 2](https://ritvik19.medium.com/papers-explained-58-palm-2-1a9a23f20d6c) | May 2023 | Successor of PALM, trained on a mixture of different pre-training objectives in order to understand different aspects of language. |
| [LIMA](https://ritvik19.medium.com/papers-explained-57-lima-f9401a5760c3) | May 2023 | A LLaMa model fine-tuned on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling. |
| [Falcon](https://ritvik19.medium.com/papers-explained-59-falcon-26831087247f) | June 2023 | An Open Source LLM trained on properly filtered and deduplicated web data alone. |
| [LLaMA 2](https://ritvik19.medium.com/papers-explained-60-llama-v2-3e415c5b9b17) | July 2023 | Successor of LLaMA. LLaMA 2-Chat is optimized for dialogue use cases. |
| [Humpback](https://ritvik19.medium.com/papers-explained-61-humpback-46992374fc34) | August 2023 | LLaMA finetuned using Instrustion backtranslation. |
| [Code LLaMA](https://ritvik19.medium.com/papers-explained-62-code-llama-ee266bfa495f) | August 2023 | LLaMA 2 based LLM for code. |
| [GPT-4V](https://ritvik19.medium.com/papers-explained-68-gpt-4v-6e27c8a1d6ea) | September 2023 | A multimodal model that combines text and vision capabilities, allowing users to instruct it to analyze image inputs. |
| [LLaMA 2 Long](https://ritvik19.medium.com/papers-explained-63-llama-2-long-84d33c26d14a) | September 2023 | A series of long context LLMs s that support effective context windows of up to 32,768 tokens. |
| [Mistral 7B](https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580) | October 2023 | Leverages grouped-query attention for faster inference, coupled with sliding window attention to effectively handle sequences of arbitrary length with a reduced inference cost. |


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
| [Conv Mixer](https://ritvik19.medium.com/papers-explained-29-convmixer-f073f0356526) | January 2022 | Processes image patches using standard convolutions for mixing spatial and channel dimensions. |

## Single Stage Object Detectors

| Paper | Date | Description |
|---|---|---|
| [SSD](https://ritvik19.medium.com/papers-explained-31-single-shot-multibox-detector-14b0aa2f5a97) | December 2015 | Discretizes bounding box outputs over a span of various scales and aspect ratios per feature map. |
| [Feature Pyramid Network](https://ritvik19.medium.com/papers-explained-21-feature-pyramid-network-6baebcb7e4b8) | December 2016 | Leverages the inherent multi-scale hierarchy of deep convolutional networks to efficiently construct feature pyramids. |
| [Focal Loss](https://ritvik19.medium.com/papers-explained-22-focal-loss-for-dense-object-detection-retinanet-733b70ce0cb1) | August 2017 | Addresses class imbalance in dense object detectors by down-weighting the loss assigned to well-classified examples. |

## Region-based Convolutional Neural Networks

| Paper | Date | Description |
|---|---|---|
| [RCNN](https://ritvik19.medium.com/papers-explained-14-rcnn-ede4db2de0ab) | November 2013 | Uses selective search for region proposals, CNNs for feature extraction, SVM for classification followed by box offset regression. |
| [Fast RCNN](https://ritvik19.medium.com/papers-explained-15-fast-rcnn-28c1792dcee0) | April 2015 | Processes entire image through CNN, employs RoI Pooling to extract feature vectors from ROIs, followed by classification and BBox regression. |
| [Faster RCNN](https://ritvik19.medium.com/papers-explained-16-faster-rcnn-a7b874ffacd9) | June 2015 | A region proposal network (RPN) and a Fast R-CNN detector, collaboratively predict object regions by sharing convolutional features. |
| [Mask RCNN](https://ritvik19.medium.com/papers-explained-17-mask-rcnn-82c64bea5261) | March 2017 | Extends Faster R-CNN to solve instance segmentation tasks, by adding a branch for predicting an object mask in parallel with the existing branch. |

## Document AI

| Paper | Date | Description |
|---|---|---|
| [Table Net](https://ritvik19.medium.com/papers-explained-18-tablenet-3d4c62269bb3) | January 2020 | An end-to-end deep learning model designed for both table detection and structure recognition. |
| [Donut](https://ritvik19.medium.com/papers-explained-20-donut-cb1523bf3281) | November 2021 | An OCR-free Encoder-Decoder Transformer model. The encoder takes in images, decoder takes in prompts & encoded images to generate the required text. |
| [DiT](https://ritvik19.medium.com/papers-explained-19-dit-b6d6eccd8c4e) | March 2022 | An Image Transformer pre-trained (self-supervised) on document images |
| [UDoP](https://ritvik19.medium.com/papers-explained-42-udop-719732358ab4) | December 2022 | Integrates text, image, and layout information through a Vision-Text-Layout Transformer, enabling unified representation. |

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

## Tabular Deep Learning

| Paper | Date | Description |
|---|---|---|
| [Entity Embeddings](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#932e) | April 2016 | Maps categorical variables into continuous vector spaces through neural network learning, revealing intrinsic properties. |
| [Wide and Deep Learning](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#bfdc) | June 2016 | Combines memorization of specific patterns with generalization of similarities. |
| [Deep and Cross Network](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#0017) | August 2017 | Combines the  a novel cross network with deep neural networks (DNNs) to efficiently learn feature interactions without manual feature engineering. |
| [Tab Transformer](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#48c4) | December 2020 | Employs multi-head attention-based Transformer layers to convert categorical feature embeddings into robust contextual embeddings. |
| [Tabular ResNet](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#46af) | June 2021 | An MLP with skip connections. |
| [Feature Tokenizer Transformer](https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#1ab8) | June 2021 | Transforms all features (categorical and numerical) to embeddings and applies a stack of Transformer layers to the embeddings. |

## Miscellaneous

| Paper | Date | Description |
|---|---|---|
| [ColD Fusion](https://ritvik19.medium.com/papers-explained-32-cold-fusion-452f33101a91) | December 2022 | A method enabling the benefits of multitask learning through distributed computation without data sharing and improving model performance. |

---

## Literature Reviewed
- [Convolutional Neural Networks](https://medium.com/dair-ai/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3)
- [Layout Transformers](https://medium.com/dair-ai/papers-explained-review-02-layout-transformers-b2d165c94ad5)
- [Region-based Convolutional Neural Networks](https://medium.com/dair-ai/papers-explained-review-03-rcnns-42c0a3974493)
- [Tabular Deep Learning](https://medium.com/dair-ai/papers-explained-review-04-tabular-deep-learning-776db04f965b)

## Reading Lists
- [Language Models](https://ritvik19.medium.com/list/language-models-11b008ddc292)
- [Layout Transformers](https://ritvik19.medium.com/list/layout-transformers-1ce4f291a9f0)
- [Object Detection](https://ritvik19.medium.com/list/object-detection-bd9e6e21ca3e)
- [RCNNs](https://ritvik19.medium.com/list/rcnns-b51467f53dc9)
- [Vision Models](https://ritvik19.medium.com/list/vision-models-61e6836230f1)
- [Document Information Processing](https://ritvik19.medium.com/list/document-information-processing-3cd900a34972)

---
Reach out to [Ritvik](https://twitter.com/RitvikRastogi19) or [Elvis](https://twitter.com/omarsar0) if you have any questions.

If you are interested to contribute, feel free to open a PR.

[Join our Discord](https://discord.gg/SKgkVT8BGJ)





