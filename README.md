![](https://user-images.githubusercontent.com/16024979/164789225-7468c77e-8816-406a-9987-44aa8d47ec47.png "Türkçe Natural Language Processing")

<div align="right">
<h6> artwork: <a href="https://en.wikipedia.org/wiki/Mihrab_(painting)">Mihrap, Osman Hamdi Bey</a>
</div>

<div align="center">
<h2><b>Turkish NLP Resources</b></h2>
Turkish NLP (Türkçe Doğal Dil İşleme) related Tools, Libraries, Models, Datasets and other resources.

<h2>Contents:</h2>
<p>
<a href="#tools--libraries">Tools & Libraries</a> |
<a href="#models">Models</a> |
<a href="#datasets">Datasets</a> |
<a href="#other-resources">Other Resources</a> |
<a href="#misc">Misc</a>
</p>
</div>
<br>

## Tools & Libraries

- [ITU Turkish NLP](http://tools.nlp.itu.edu.tr/api_usage.jsp) (Web Based & API) : Tools of Istanbul Technical University, Natural Language Processing Group.
- [spaCy Turkish models](https://huggingface.co/turkish-nlp-suite) : Pre-trained Turkish models for spaCy.
- [VNLP](https://github.com/vngrs-ai/vnlp) (Python) : State-of-the-art, lightweight NLP tools for Turkish. [![][repo]](https://github.com/vngrs-ai/vnlp)
- [TDD - Tools](https://tools.tdd.ai/) (Web Based) : Online tools provided by the Turkish Data Depository (TDD) project.
- [Zemberek-NLP](https://github.com/ahmetaa/zemberek-nlp) (Java) : Natural Language Processing library for Turkish. [![][repo]](https://github.com/ahmetaa/zemberek-nlp)
- [Zemberek-Python](https://github.com/Loodos/zemberek-python) (Python) : Python implementation of Zemberek. [![][repo]](https://github.com/Loodos/zemberek-python)
- [Zemberek-Server](https://github.com/cbilgili/zemberek-nlp-server) (Docker) : REST Docker server based on Zemberek Turkish NLP library. [![][repo]](https://github.com/cbilgili/zemberek-nlp-server)
- [TRmorph](https://github.com/coltekin/TRmorph) (FST) : A finite-state morphological analyzer for Turkish. [![][repo]](https://github.com/coltekin/TRmorph)
- [Mukayese](https://github.com/alisafaya/mukayese) (Python) : Benchmarking platform for various Turkish NLP tools and tasks. [![][repo]](https://github.com/alisafaya/mukayese)
- [SadedeGel](https://github.com/GlobalMaksimum/sadedegel) (Python) : Unsupervised extraction-based news summarization tool. [![][repo]](https://github.com/GlobalMaksimum/sadedegel)
- [Turkish Stemmer](https://github.com/otuncelli/turkish-stemmer-python/) (Python) : Stemmer algorithm for Turkish language. [![][repo]](https://github.com/otuncelli/turkish-stemmer-python/)
- [sinKAF](https://github.com/eonurk/sinkaf) (Python) : An ML library for profanity detection in Turkish sentences. [![][repo]](https://github.com/eonurk/sinkaf)
- [TrTokenizer](https://github.com/apdullahyayik/TrTokenizer) (Python) : Sentence and word tokenizers for the Turkish language. [![][repo]](https://github.com/apdullahyayik/TrTokenizer)
- Tools by [Starlang](https://github.com/StarlangSoftware) (Multi/Python) : [Morphological Analysis](https://github.com/StarlangSoftware/TurkishMorphologicalAnalysis-Py), [Spell Checker](https://github.com/StarlangSoftware/TurkishSpellChecker-Py), [Dependency Parser](https://github.com/StarlangSoftware/TurkishDependencyParser-Py), [Deasciifier](https://github.com/StarlangSoftware/TurkishDeasciifier-Py), [NER](https://github.com/StarlangSoftware/TurkishNamedEntityRecognition-Py).
- [snnclsr/NER](https://github.com/snnclsr/ner) (Python) : Named Entity Recognition system for the Turkish language. [![][repo]](https://github.com/snnclsr/ner)

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Models

- [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased) : Turkish BERT/DistilBERT, ELECTRA and ConvBERT models. [![][repo]](https://github.com/stefan-it/turkish-bert)
- [ELMO For ManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs) : Pre-trained ELMo Representations for Many Languages.
- [Fasttext - Word Vector](https://fasttext.cc/docs/en/crawl-vectors.html) : Pre-trained word vectors for 157 languages, trained on Common Crawl and Wikipedia using fastText.
- [Loodos/Turkish Language Models](https://github.com/Loodos/turkish-language-models) : In this repository, we publish Transformer based Turkish language models and related tools.
- [Hugging Face - Models/Turkish](https://huggingface.co/models?search=turkish)

### Word Embeddings

- [Floret Embeddings](https://huggingface.co/turkish-nlp-suite) : Turkish Floret Embeddings, large and medium sized.
- [VNLP Word Embeddings](https://vnlp.readthedocs.io/en/latest/main_classes/word_embeddings.html) : Word2Vec Turkish word embeddings.
- [TurkishGloVe](https://github.com/inzva/Turkish-GloVe) : Turkish GloVe word embeddings.

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Datasets

- [TDD - Türkçe Dil Deposu (Turkish Language Repository)](https://data.tdd.ai/) : The Turkish Natural Language Processing Project aims to prepare foundational datasets for processing Turkish texts.
- [ITU NLP Group - Datasets](http://tools.nlp.itu.edu.tr/Datasets) : Datasets from the Istanbul Technical University Natural Language Processing Group.
- [Boğaziçi University TABI - NLI-TR](https://github.com/boun-tabi/NLI-TR) : Natural Language Inference datasets translated into Turkish using Amazon Translate. [![][repo]](https://github.com/boun-tabi/NLI-TR)
- [Turkish NLP Suite Datasets](https://github.com/turkish-nlp-suite) : Diverse linguistic resources including NER, medical NLP, and sentiment analysis datasets. [![][repo]](https://github.com/turkish-nlp-suite)

### Multilingual Datasets:

- [Amazon MASSIVE](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding) : MASSIVE is a parallel dataset of 1M utterances across 51 languages with annotations for the NLU tasks of intent prediction and slot annotation. [![][repo]](https://github.com/alexa/massive)
- [OPUS: en-tr](https://opus.nlpl.eu/index.php?src=en&trg=tr) : OPUS is a growing collection of translated texts from the web. In the OPUS project we try to convert and align free online data, to add linguistic annotation, and to provide the community with a publicly available parallel corpus.
- [CC-100](https://data.statmt.org/cc-100/) : Monolingual Datasets from Web Crawl Data. This corpus comprises of monolingual data for 100+ languages.
- [OSCAR](https://oscar-corpus.com/) : is a huge multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the Ungoliant architecture.

### Treebank:

- [Universal Dependencies](https://universaldependencies.org/#turkish-treebanks) : is an international cooperative project to create treebanks of the world's languages. The project seeks to develop cross-linguistically consistent treebank annotation of morphology and syntax for multiple languages. [![][repo]](https://github.com/UniversalDependencies)
- [UD Turkish Kenet](https://universaldependencies.org/treebanks/tr_kenet/index.html) Turkish-Kenet UD Treebank consists of 18,700 manually annotated sentences and 178,700 tokens. Its corpus consists of dictionary examples from TDK. [![][repo]](https://github.com/StarlangSoftware/TurkishWordNet)
- [UD Turkish BOUN](https://universaldependencies.org/treebanks/tr_boun/index.html) : BOUN Treebank is created by the TABILAB and supported by TÜBİTAK. This corpus contains 9761 sentences, 121,214 tokens. [![][repo]](https://github.com/boun-tabi/UD_Turkish-BOUN)

### Other Data:

- [Turkish Song Lyrics (Türkçe Şarkı Sözleri)](https://www.kaggle.com/datasets/emreokcular/turkish-song-lyrics)
- [Turkish Folk Song Lyrics (Türkçe Türkü Sözleri)](https://www.kaggle.com/datasets/emreokcular/turkish-folk-song-lyrics)
- [Turkish Poems (Türkçe Şiirler)](https://www.kaggle.com/datasets/emreokcular/turkish-poems)
- [Turkish Idioms and Proverbs (Türkçe Atasözleri ve Deyimler)](https://www.kaggle.com/datasets/emreokcular/turkish-idioms-and-proverbs)
- [hermitdave/Frequency Word List](https://github.com/hermitdave/FrequencyWords)
- [Fırat University - Veri Setleri](http://buyukveri.firat.edu.tr/veri-setleri/)
- [Bilkent Turkish Writings Dataset](https://github.com/selimfirat/bilkent-turkish-writings-dataset)
- [170k Turkish Sentences from Wikipedia](https://www.kaggle.com/datasets/mahdinamidamirchi/turkish-sentences-dataset)
- [Wiktionary: Frequency Lists - Turkish](https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists#Turkish)
- [ooguz/Bad Word Blacklist for Turkish](https://github.com/ooguz/turkce-kufur-karaliste)
- [ahmetax/Turkish Stop Words List](https://github.com/ahmetax/trstop)
- [NLTK - Stop Words](https://www.nltk.org/book/ch02.html#wordlist-corpora)
- [Tatoeba: Multilingual Sentences](https://tatoeba.org/tr/downloads)
- [466k English Words](https://github.com/dwyl/english-words)

### Other Sources:

- [Google Dataset Search/Turkish](https://datasetsearch.research.google.com/search?src=0&query=turkish)
- [Kaggle - Datasets/Turkish](https://www.kaggle.com/search?q=turkish+in:datasets)
- [Hugging Face - Datasets/Turkish](https://huggingface.co/datasets?search=turkish)

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Other Resources

### Books:

- [Turkish Natural Language Processing (Theory and Applications of Natural Language Processing)](https://www.amazon.com/Turkish-Natural-Language-Processing-Applications/dp/331990163X)

### Videos:

- [BOUN - Yapay Öğrenmeye Giriş - İsmail Arı Yaz Okulu 2018](https://www.youtube.com/playlist?list=PLMGWwuh6-mEcTODbE22Q1KATHeZYAQTTg)
- [BOUN - Doğal Dil İşleme - İsmail Arı Yaz Okulu 2018](https://www.youtube.com/playlist?list=PLMGWwuh6-mEe57iOtf1uo5DgZB288l0CQ)
- [BOUN - Konuşma / İşleme - İsmail Arı Yaz Okulu 2018](https://www.youtube.com/playlist?list=PLMGWwuh6-mEdAkjnSLbyUq7Ca21UiSi6F)
- [BOUN - Yapay Öğrenme Yaz Okulu 2020](https://www.youtube.com/playlist?list=PLMGWwuh6-mEfmMAUoQZNfEA51RGh7bMyh)
- [Açık Seminer - NLP 101 Doğal Dil İşlemeye Giriş ve Uygulamalı Metin Madenciliği](https://www.youtube.com/watch?v=tm1K9ZvJXJI)
- [Starlang Yazılım Channel](https://www.youtube.com/@starlangyazilim/videos)
- [NLP with Duygu](https://www.youtube.com/@NLPwithDuygu)

### Articles:

- [Türkçe ve Doğal Dil İşleme](https://dergipark.org.tr/tr/download/article-file/207207)
- [Türkçe Tweetler Üzerinde Otomatik Soru Tespiti](https://dergipark.org.tr/tr/download/article-file/605454)
- [Classification of News according to Age Groups Using NLP](https://dergipark.org.tr/tr/download/article-file/1140110)
- [Açık Kaynak Doğal Dil İşleme Kütüphaneleri](https://dergipark.org.tr/tr/download/article-file/1573501)
- [Neden yasaklandı? Depremle ilgili Ekşi Sözlük yorumlarına NLP gözüyle bakış](https://medium.com/p/ce65ece62aea)
- [A collection of brand new datasets for Turkish NLP](https://medium.com/p/fc83ca3c95df)

### Sample Notebooks/Snippets:

- [kodiks/Turkish News Category Classification Tutorial](https://github.com/kodiks/turkish-news-classification)
- [ezgisubasi/Turkish Tweets Sentiment Analysis](https://github.com/ezgisubasi/turkish-tweets-sentiment-analysis)
- [merveenoyan/NLP için Derlediğim Fonksiyonlar](https://github.com/merveenoyan/NLP-Helpers)

### Blog Posts:

- [Doğal Dil İşleme (Natural Language Processing)](https://merveenoyan.medium.com/do%C4%9Fal-dil-i%CC%87%C5%9Fleme-natural-language-processing-2d7c72daf245)
- [Bir Veri Bilimcinin Araç Çantası](https://merveenoyan.medium.com/bir-veri-bilimcinin-ara%C3%A7-%C3%A7antas%C4%B1-ca51fb5cd19e)
- [Doğal Dil İşleme Metin Temsili Yöntemleri](https://qann.medium.com/metin-temsili-nedir-nlpde-metinleri-say%C4%B1lara-d%C3%B6n%C3%BC%C5%9Ft%C3%BCrmenin-5-temel-yolu-860859b2cc09)

### Other Lists:

- [Açık Veri Kaynakları](https://github.com/kaymal/acik-veri) : List of open data sources in Turkey (Official Institutions, Municipalities, Universities, etc.). [![][repo]](https://github.com/kaymal/acik-veri)
- [Awesome Turkish NLP](https://github.com/yusufusta/awesome-turkish-nlp) : A curated list of Turkish NLP resources. [![][repo]](https://github.com/yusufusta/awesome-turkish-nlp)
- [Türkçe Yapay Zeka Kaynakları](https://github.com/deeplearningturkiye/turkce-yapay-zeka-kaynaklari) : Collection of AI resources in Turkish. [![][repo]](https://github.com/deeplearningturkiye/turkce-yapay-zeka-kaynaklari)

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Misc

- [Kip](https://kip-dili.github.io/) : A programming language in Turkish where grammatical case and mood are part of the type system. [![][repo]](https://github.com/kip-dili/kip)

<div align="right">
    <b><a href="#contents">↥ Back To Top</a></b>
</div>

## Contributing

Your contributions are welcome! If you want to contribute to this list, send a _pull request_ or just open a _new issue_.

[repo]: https://raw.githubusercontent.com/agmmnn/awesome-blender/master/imgs/github.svg
