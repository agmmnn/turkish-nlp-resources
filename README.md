![](https://user-images.githubusercontent.com/16024979/164789225-7468c77e-8816-406a-9987-44aa8d47ec47.png "Türkçe Natural Language Processing")

<div align="right">
<i><h6> artwork: <a href="https://en.wikipedia.org/wiki/Mihrab_(painting)">Mihrap, Osman Hamdi Bey</a></h6></i>
</div>

<div align="center">
<h2><b>Turkish NLP Resources</b></h2>
Turkish NLP (Türkçe Doğal Dil İşleme) Tools, Libraries, Models, Datasets, and other resources.
<br>
<i>Aligned with new NLP Trends: Generative AI, Retrieval Systems, and Evaluation</i>

<h2>Contents:</h2>
<p> |
<a href="#generative-ai--llms">Generative AI & LLMs</a> |
<a href="#retrieval--semantic-search-rag">Retrieval & RAG</a> |
<a href="#evaluation--benchmarks">Evaluation & Benchmarks</a> |
<a href="#encoder-models">Encoder Models</a> |
<a href="#tools--libraries">Tools & Libraries</a> |
<a href="#datasets">Datasets</a> |
<a href="#community--learning">Community & Learning</a> |
<a href="#misc">Misc</a> |
</p>
</div>
<br>

## Generative AI & LLMs

### Foundation & Chat Models

> _Language models specific to Turkish, ranging from adaptations of open weights (Llama, Mistral) to native pretrained models._

- [Trendyol LLMs](https://huggingface.co/Trendyol/models) : Bilingual (TR/EN) models ranging from 7B to 70B parameters, including specialized cybersecurity variants.
- [Kumru-2B](https://huggingface.co/vngrs-ai/Kumru-2B) : Decoder-only foundational models trained from scratch for Turkish with a native tokenizer. [blog](https://medium.com/vngrs/kumru-llm-34d1628cfd93)
- [TURNA](https://huggingface.co/boun-tabi-LMG/TURNA) : A 1.1B parameter foundational model for NLU and generation.
- [Cosmos Turkish Llama](https://huggingface.co/ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1) : The Cosmos Llama is designed for text generation tasks, trained with DPO for coherent Turkish continuation.
- [Kanarya-2b](https://huggingface.co/asafaya/kanarya-2b) : Turkish GPT-J model trained on large-scale corpora.
- [Turkcell-LLM-7b-v1](https://huggingface.co/TURKCELL/Turkcell-LLM-7b-v1) : Extended version of Mistral fine-tuned on Turkish instruction sets.
- [WiroAI/wiroai-turkish-llm-9b](https://huggingface.co/WiroAI/wiroai-turkish-llm-9b) : Robust language models adapted to Turkish culture and context.
- [Kocdigital-LLM-8b-v0.1](https://huggingface.co/KOCDIGITAL/Kocdigital-LLM-8b-v0.1) : Fine-tuned version of Llama3 8b for Turkish.

### Domain Specific LLMs

> _Models adapted for specific verticals (Legal, Medical, Finance)._

- [Mecellem](https://huggingface.co/collections/newmindai/mecellem-models) : Specialized ModernBERT-based models for the Turkish legal domain. [arxiv](https://arxiv.org/abs/2601.16018)

### LLM Integrations (MCP Servers)

> _Model Context Protocol (MCP) servers enabling AI agents to interact with Turkish data sources._

- [Borsa MCP](https://github.com/saidsurucu/borsa-mcp) : Istanbul Stock Exchange (BIST) and investment fund data.
- [Yargı MCP](https://github.com/saidsurucu/yargi-mcp) : Search for Turkish Legal Databases (Yargıtay, Danıştay).
- [Mevzuat MCP](https://github.com/saidsurucu/mevzuat-mcp) : Search Turkish Legislation (laws, regulations).
- [YÖK Tez MCP](https://github.com/saidsurucu/yoktez-mcp) : Turkish National Thesis Center (YÖK Tez) search.
- [YÖK Atlas MCP](https://github.com/saidsurucu/yokatlas-mcp) : YÖK Atlas higher education and ranking data.

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Retrieval & Semantic Search (RAG)

> _Crucial for RAG (Retrieval Augmented Generation) pipelines, moving beyond keyword search._

### Late-Interaction Models

> _Late-interaction models (ColBERT) are specifically designed for high-performance retrieval tasks._

- [TurkColBERT](https://huggingface.co/collections/newmindai/turkcolbert-turkish-late-interaction-models) : Benchmark and collection of token-level matching models for high-performance retrieval. [arxiv](https://arxiv.org/abs/2511.16528), [blog](https://huggingface.co/blog/newmindai/late-interaction-models)

### Embedding Models

> _Embedding models for semantic search and retrieval._

- [TurkEmbed4Retrieval](https://huggingface.co/newmindai/TurkEmbed4Retrieval) : Specialized embedding model for Turkish retrieval tasks.
- [Mursit-Large-TR-Retrieval](https://huggingface.co/newmindai/Mursit-Large-TR-Retrieval) : Late-interaction retrieval model for Turkish.
- [TY-ecomm-embed-multilingual-base-v1.2.0](https://huggingface.co/Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0) : Multilingual e-commerce embeddings.
- [Floret Embeddings](https://huggingface.co/turkish-nlp-suite) : Turkish Floret Embeddings, large and medium sized.
- [VNLP Word Embeddings](https://vnlp.readthedocs.io/en/latest/main_classes/word_embeddings.html) : Word2Vec Turkish word embeddings.
- [TurkishGloVe](https://github.com/inzva/Turkish-GloVe) : Turkish GloVe word embeddings.

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Evaluation & Benchmarks

> _Leaderboards and datasets to validate model performance in Turkish._

- [Mezura](https://huggingface.co/spaces/newmindai/Mezura) : Leaderboard focusing on human evaluation (ELO) and RAG performance.
- [Mizan](https://huggingface.co/spaces/newmindai/Mizan) : Embedding model leaderboard for retrieval and clustering tasks.
- [TurkBench](https://huggingface.co/spaces/TurkBench/TurkBench) : Comprehensive generative LLM benchmark with 21 subtasks. [arxiv](https://arxiv.org/abs/2601.07020)
- [Cetvel](https://huggingface.co/spaces/KUIS-AI/Cetvel) : A 26-task benchmark including translation, summarization, and correction.
- [TR-MMLU](https://github.com/NovusResearch/TR-MMLU) : Evaluation framework with 6,200 Turkish-specific multiple-choice questions.
- [TrGLUE](https://github.com/turkish-nlp-suite/TrGLUE) : Turkish-native corpora curated for GLUE-style evaluations.

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Encoder Models

> _Traditional Transformer models (BERT, RoBERTa, etc.) and Word Vectors._

- [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased) : Turkish BERT/DistilBERT, ELECTRA and ConvBERT models.
- [TurkishBERTweet](https://huggingface.co/VRLLab/TurkishBERTweet) : A BERTweet model fine-tuned on Turkish tweets.
- [Loodos/Turkish Language Models](https://github.com/Loodos/turkish-language-models) : Transformer based Turkish language models.
- [ELMO For ManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs) : Pre-trained ELMo Representations.
- [Fasttext - Word Vector](https://fasttext.cc/docs/en/crawl-vectors.html) : Pre-trained word vectors for 157 languages.

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Tools & Libraries

> _Core libraries for morphological analysis, tokenization, and processing._

- [VNLP](https://github.com/vngrs-ai/vnlp) (Python) : State-of-the-art, lightweight NLP tools for Turkish.
- [Zemberek-NLP](https://github.com/ahmetaa/zemberek-nlp) (Java) : The veteran NLP library for Turkish (Morphology, Spell Check, etc.).
- [Zemberek-Python](https://github.com/Loodos/zemberek-python) (Python) : Python wrapper/implementation of Zemberek.
- [Zemberek-Server](https://github.com/cbilgili/zemberek-nlp-server) (Docker) : REST Docker server for Zemberek.
- [TRmorph](https://github.com/coltekin/TRmorph) (FST) : Finite-state morphological analyzer.
- [spaCy Turkish models](https://huggingface.co/turkish-nlp-suite) : Pre-trained Turkish pipelines for spaCy.
- [Starlang Tools](https://github.com/StarlangSoftware) (Python) : Comprehensive suite (Morphology, Spell Check, Dependency Parsing, Deasciifier, NER).
- [ITU Turkish NLP](http://tools.nlp.itu.edu.tr/api_usage.jsp) (Web/API) : Tools from ITU Natural Language Processing Group.
- [Nuve](https://github.com/hrzafer/nuve) (C#) : Turkish NLP library for morphological analysis.
- [SadedeGel](https://github.com/GlobalMaksimum/sadedegel) (Python) : Extraction-based news summarization.
- [Turkish Stemmer](https://github.com/otuncelli/turkish-stemmer-python/) (Python) : Stemming algorithm.
- [sinKAF](https://github.com/eonurk/sinkaf) (Python) : Profanity detection library.
- [TrTokenizer](https://github.com/apdullahyayik/TrTokenizer) (Python) : Sentence and word tokenizers.
- [snnclsr/NER](https://github.com/snnclsr/ner) (Python) : Named Entity Recognition system.
- [Helsinki-NLP Translation](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr) : Neural machine translation (EN-TR).

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Datasets

> _Extensive corpora and collections for training and evaluation._

### Instruction Tuning & Dialogue (LLM)

- [InstrucTurca](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca) : 2.58M instruction samples (OpenOrca/MedText translations).
- [Turkish-Alpaca](https://huggingface.co/datasets/TFLai/Turkish-Alpaca) : 52k cleaned/verified instruction following samples.
- [WikiRAG-TR](https://huggingface.co/datasets/Metin/WikiRAG-TR) : Questions derived from Turkish Wikipedia for RAG.
- [turkish-math-186k](https://huggingface.co/datasets/ituperceptron/turkish-math-186k) : Large-scale dataset for mathematical reasoning.
- [Boğaziçi University TABI - NLI-TR](https://github.com/boun-tabi/NLI-TR) : Natural Language Inference datasets.

### Multimodal & Vision

- [TurkishLLaVA OCR Enhancement](https://huggingface.co/datasets/ytu-ce-cosmos/turkce-kitap) : Specialized books collection for OCR improvement.
- [unsloth-pmc-vqa-tr](https://huggingface.co/datasets/nezahatkorkmaz/unsloth-pmc-vqa-tr) : Turkish PMC-VQA (Medical Visual Question Answering).
- [BosphorusSign22k](https://ogulcanozdemir.github.io/bosphorussign22k/) : Turkish Sign Language Recognition (SLR) benchmark.

### Major Corpora & Collections

- [Cosmos Datasets](https://huggingface.co/ytu-ce-cosmos/datasets) : Extensive datasets from YTU Cosmos Research Group.
- [Trendyol Datasets](https://huggingface.co/Trendyol/datasets) : E-commerce and general datasets from Trendyol.
- [Turkish National Corpus (TNC)](https://www.tnc.org.tr/) : Balanced, large scale (50M words) general-purpose corpus.
- [TS Corpus](https://tscorpus.com/) : Independent project for Turkish corpora and datasets.
- [TDD - Turkish Data Depository](https://data.tdd.ai/) : Foundational datasets.
- [METU Corpora](https://ii.metu.edu.tr/metu-corpora-research-group) : MTC and Discourse Bank.

### Treebanks (Syntax & Morphology)

- [Universal Dependencies (UD)](https://universaldependencies.org/#turkish-treebanks) : Standardized cross-linguistic treebanks.
- [UD Turkish BOUN](https://universaldependencies.org/treebanks/tr_boun/index.html) : 9.7k sentences, created by TABILAB.
- [UD Turkish Kenet](https://universaldependencies.org/treebanks/tr_kenet/index.html) : 18.7k sentences, based on TDK dictionary.
- [UD Ottoman Turkish](https://github.com/UniversalDependencies/UD_Ottoman_Turkish-BOUN) : Historical treebank.
- [METU-Sabancı Treebank](https://web.itu.edu.tr/gulsenc/treebanks.html) : Syntactic analysis resources.

### Sentiment, General NLP & Others

- [SentiTurca](https://huggingface.co/datasets/turkish-nlp-suite/SentiTurca) : Sentiment analysis benchmark.
- [FSMTSAD](https://github.com/kevserbusrayildirim/FSMTSAD) : Balanced sentiment dataset (Hotel, Movie, Product).
- [HisTR](https://huggingface.co/datasets/Saziye/HisTR) : NER dataset for historical Turkish.
- [Turkish NLP Suite Datasets](https://github.com/turkish-nlp-suite) : NER, medical, and sentiment resources.
- [Amazon MASSIVE](https://github.com/alexa/massive) & [OPUS](https://opus.nlpl.eu/) : Multilingual resources.
- [Common Crawl (CC-100)](https://data.statmt.org/cc-100/) & [OSCAR](https://oscar-corpus.com/) : Large/Web-scale corpora.
- **Miscellaneous**: [Song Lyrics](https://www.kaggle.com/datasets/emreokcular/turkish-song-lyrics), [Poems](https://www.kaggle.com/datasets/emreokcular/turkish-poems), [Idioms](https://www.kaggle.com/datasets/emreokcular/turkish-idioms-and-proverbs), [Stop Words](https://github.com/ahmetax/trstop), [Bad Word Blacklist](https://github.com/ooguz/turkce-kufur-karaliste), [Tatoeba: Multilingual Sentences](https://tatoeba.org/tr/downloads)

### Dataset Search

- [Google Dataset Search/Turkish](https://datasetsearch.research.google.com/search?src=0&query=turkish)
- [Kaggle - Datasets/Turkish](https://www.kaggle.com/search?q=turkish+in:datasets)
- [Hugging Face - Datasets/Turkish](https://huggingface.co/datasets?search=turkish)

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Community & Learning

### YouTube Channels

- [KUIS AI](https://www.youtube.com/@kuisaicenter/videos)
- [Türkiye Yapay Zeka İnisiyatifi](https://www.youtube.com/c/T%C3%BCrkiyeYapayZeka%C4%B0nisiyatifi)
- [Trendyol Tech](https://www.youtube.com/@TrendyolTech/videos)
- [Starlang Yazılım](https://www.youtube.com/@starlangyazilim/videos)
- [NLP with Duygu](https://www.youtube.com/@NLPwithDuygu)

### Awesome Lists

- [Awesome Turkish NLP](https://github.com/yusufusta/awesome-turkish-nlp) : Alternative curated list.
- [Awesome Turkish Language Models](https://github.com/kesimeg/awesome-turkish-language-models) : Curated list of models.
- [Açık Veri Kaynakları](https://github.com/kaymal/acik-veri) : Open data sources in Turkey.

### Educational Resources

- [Turkish Natural Language Processing - Kemal Oflazer](https://www.amazon.com/Turkish-Natural-Language-Processing-Applications/dp/331990163X)

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Misc

- [Kip](https://kip-dili.github.io/) : A programming language in Turkish based on case and mood.

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Contributing

Your contributions are welcome! If you want to contribute to this list, send a _pull request_ or just open a _new issue_.
