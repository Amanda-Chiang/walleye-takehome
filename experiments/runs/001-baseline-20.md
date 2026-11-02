# Error report: `results/baseline-20`

| Outcome | Count |
| --- | ---: |
| retrieval_miss | 9 |
| missed_abstain | 4 |
| retrieved_answered | 3 |
| retrieval_partial | 2 |
| correct_abstain | 1 |
| over_abstain | 1 |

## Questions

### Q0 `1912.04961`: retrieval_miss

- **Question:** What is the baseline?
- **Prediction:** For Dosage extraction, the baseline is “Nearest Number”: predict the number nearest to the Medication Name, or “none” if no number is mentioned or the Medication Name is not detected. For Frequency extraction, it is “Random Top-3”: predict a random tag among “none,” “daily,” and “twice a day.” (Paper 1912.04961)
- **Gold:** QA PGNet, Multi-decoder QA PGNet with lookup table embedding
- **Gold:** QA PGNet and Multi-decoder QA PGNet
- **Retrieved:** ['p00033', 'p00047', 'p00013', 'p00000', 'p00014']; **cited:** ['p00033']; **gold evidence (best of 2 references):** ['p00032']
- **Missed** `p00032` [Experiments ::: Model variations]: We consider QA PGNet and Multi-decoder QA PGNet with lookup table embedding as baseline models and improve on the baselines with other variations described below.

### Q1 `1909.09551`: retrieval_partial

- **Question:** How they utilize LDA and Gibbs sampling to evaluate ISWC and WWW publications?
- **Prediction:** In paper 1909.09551, they applied LDA with Gibbs sampling to ISWC and WWW publications from 2013–2017, using titles and abstracts from DBLP. They implemented inference with MALLET to obtain topic models, then examined topic trends and relationships between LDA topics and paper features, including researchers’ interests.
- **Gold:** the LDA approaches to recommendation systems and given the importance of research, we have studied recent impressive articles on this subject and presented a taxonomy of recommendation systems based on LDA of the recent research, we evaluated ISWC and WWW conferences articles from DBLP website and used the Gibbs sampling algorithm as an evaluation parameter
- **Gold:** discover the trends of the topics and find relationship between LDA topics and paper features and generate trust tags,  learn a LDA model with 100 topics; $\alpha =0.01$, $\beta = 0.01$ and using Gibbs sampling as a parameter estimation
- **Retrieved:** ['p00061', 'p00066', 'p00055', 'p00001', 'p00005']; **cited:** ['p00066', 'p00055', 'p00061']; **gold evidence (best of 2 references):** ['p00011', 'p00061']
- **Missed** `p00011` [Natural language Processing and LDA topic model ::: Gibbs sampling and Learning LDA]: As previously mentioned, Topic modeling can find a collection of distributions over words for each topic and the relationship of topics with each document. To perform approximate inference and learning LDA, there are many inference methods 

### Q2 `2002.07306`: retrieval_miss

- **Question:** How much training data from the non-English language is used by the system?
- **Prediction:** Unanswerable
- **Gold:** No data. Pretrained model is used.
- **Retrieved:** ['p00003', 'p00022', 'p00043', 'p00023', 'p00000']; **cited:** []; **gold evidence (best of 1 references):** ['p00004']
- **Missed** `p00004` [Introduction]: Do multilingual models always need to be trained from scratch? Can we transfer linguistic knowledge learned by English pre-trained models to other languages? In this work, we develop a technique to rapidly transfer an existing pre-trained m

### Q3 `1806.04511`: retrieval_miss

- **Question:** which non-english language was the had the worst results?
- **Prediction:** Unanswerable
- **Gold:** Turkish
- **Retrieved:** ['p00026', 'p00025', 'p00023', 'p00019', 'p00027']; **cited:** []; **gold evidence (best of 1 references):** ['p00028']
- **Missed** `p00028` [Discussion]: Considering the improvements over the majority baseline achieved by the RNN model for both non-English (on the average 22.76% relative improvement; 15.82% relative improvement on Spanish, 72.71% vs. 84.21%, 30.53% relative improvement on Tu

### Q4 `1705.01265`: retrieved_answered

- **Question:** Do they report results only on English datasets?
- **Prediction:** Yes. Paper 1705.01265 reports using English tweets for its experimental study.
- **Gold:** Yes
- **Gold:** Yes
- **Retrieved:** ['p00008', 'p00021', 'p00032', 'p00011', 'p00028']; **cited:** ['p00008']; **gold evidence (best of 2 references):** ['p00008']

### Q5 `2001.01269`: retrieval_miss

- **Question:** How are the supervised scores of the words calculated?
- **Prediction:** The paper says each word’s supervised polarity score is calculated from the training data using a formula shown as (DISPLAY_FORM4), but the formula itself is not provided in the passages. In the dictionary approach, that score is multiplied by each value in the word’s row.
- **Gold:** (+1 or -1), words of opposite polarities (e.g. “happy" and “unhappy") get far away from each other
- **Retrieved:** ['p00024', 'p00022', 'p00026', 'p00027', 'p00050']; **cited:** ['p00022']; **gold evidence (best of 1 references):** ['p00025']
- **Missed** `p00025` [Methodology ::: Dictionary-based Approach]: The effect of this multiplication is exemplified in Figure FIGREF7, showing the positions of word vectors in the VSM. Those “x" words are sentimentally negative words, those “o" words are sentimentally positive ones. On the top coordinate p

### Q6 `1908.07195`: missed_abstain

- **Question:** Is the discriminator's reward made available at each step to the generator?
- **Prediction:** No. The passages say the discriminator supplies reward signals, but do not establish that its reward is made available to the generator at each step.
- **Gold:** Unanswerable
- **Retrieved:** ['p00008', 'p00030', 'p00032', 'p00041', 'p00013']; **cited:** ['p00032']; **gold evidence (best of 1 references):** []

### Q7 `1910.12574`: retrieved_answered

- **Question:** What baseline is used?
- **Prediction:** Paper 1910.12574 uses Waseem and Hovy, Davidson et al., and Waseem et al. as baselines.
- **Gold:** Waseem and Hovy BIBREF5, Davidson et al. BIBREF9, Waseem et al. BIBREF10
- **Gold:** Waseem and Hovy BIBREF5, Davidson et al. BIBREF9, and Waseem et al. BIBREF10
- **Retrieved:** ['p00024', 'p00023', 'p00014', 'p00020', 'p00031']; **cited:** ['p00024']; **gold evidence (best of 2 references):** ['p00024']

### Q8 `1611.08661`: retrieval_miss

- **Question:** What baselines are used for comparison?
- **Prediction:** Unanswerable
- **Gold:** TransE
- **Gold:** TransE
- **Retrieved:** ['p00009', 'p00048', 'p00050', 'p00042', 'p00076']; **cited:** []; **gold evidence (best of 2 references):** ['p00054']
- **Missed** `p00054` [Link Prediction]: Experimental results on both WN18 and FB15k are shown in Table 2 , where we use “Jointly(CBOW)”, “Jointly(LSTM)” and “Jointly(A-LSTM)” to represent our jointly encoding models with CBOW, LSTM and attentive LSTM text encoders. Our baseline i

### Q9 `1705.01214`: retrieval_miss

- **Question:** What is the state of the art described in the paper?
- **Prediction:** The paper says there is little research on chatbot engines integrating coordination laws with natural language, and describes its architecture as the first designed to support coordinated-aware chatbots in multiparty conversations.
- **Gold:** ELIZA,  PARRY, A.L.I.C.E., Cleverbot
- **Retrieved:** ['p00016', 'p00156', 'p00042', 'p00290', 'p00008']; **cited:** ['p00042']; **gold evidence (best of 1 references):** ['p00017', 'p00018', 'p00019', 'p00020', 'p00021']
- **Missed** `p00017` [Conversational Systems]: In this section we discuss the state of the art on conversational systems in three perspectives: types of interactions, types of architecture, and types of context reasoning. Then we present a table that consolidates and compares all of the
- **Missed** `p00018` [Conversational Systems]: ELIZA BIBREF11 was one of the first softwares created to understand natural language processing. Joseph Weizenbaum created it at the MIT in 1966 and it is well known for acting like a psychotherapist and it had only to reflect back onto pat
- **Missed** `p00019` [Conversational Systems]: Right after ELIZA came PARRY, developed by Kenneth Colby, who is psychiatrist at Stanford University in the early 1970s. The program was written using the MLISP language (meta-lisp) on the WAITS operating system running on a DEC PDP-10 and 
- **Missed** `p00020` [Conversational Systems]: A.L.I.C.E. (Artificial Linguistic Internet Computer Entity) BIBREF12 appeared in 1995 but current version utilizes AIML, an XML language designed for creating stimulus-response chat robots BIBREF13 . A.L.I.C.E. bot has, at present, more tha
- **Missed** `p00021` [Conversational Systems]: Cleverbot (1997-2014) is a chatbot developed by the British AI scientist Rollo Carpenter. It passed the 2011 Turing Test at the Technique Techno-Management Festival held by the Indian Institute of Technology Guwahati. Volunteers participate

### Q10 `2002.01320`: correct_abstain

- **Question:** How big is Augmented LibriSpeech dataset?
- **Prediction:** Unanswerable
- **Gold:** Unanswerable
- **Gold:** Unanswerable
- **Retrieved:** ['p00001', 'p00004', 'p00003', 'p00002', 'p00012']; **cited:** []; **gold evidence (best of 2 references):** []

### Q11 `2002.07306`: retrieval_miss

- **Question:** What metrics are used for evaluation?
- **Prediction:** Accuracy, LAS (Labeled Attachment Score), and UAS (Unlabeled Attachment Score) are used for evaluation (paper 2002.07306).
- **Gold:** translation probabilities, Labeled Attachment Scores (LAS)
- **Gold:** accuracy, Labeled Attachment Scores (LAS)
- **Retrieved:** ['p00052', 'p00053', 'p00054', 'p00044', 'p00043']; **cited:** ['p00053', 'p00054']; **gold evidence (best of 2 references):** ['p00026', 'p00038']
- **Missed** `p00026` [Zero-shot Experiments ::: Estimating translation probabilities]: Since pre-trained models operate on subword level, we need to estimate subword translation probabilities. Therefore, we subsample 2M sentence pairs from each parallel corpus and tokenize the data into subwords before running fast-align BIBR
- **Missed** `p00038` [Results ::: Universal Dependency Parsing]: We build on top of RAMEN a graph-based dependency parser BIBREF27. For the purpose of evaluating the contextual representations learned by our model, we do not use part-of-speech tags. Contextualized representations are directly fed into De

### Q12 `2002.01207`: retrieval_miss

- **Question:** what surface-level features are used?
- **Prediction:** The paper uses word surface forms, stems, prefixes, and suffixes (including noun suffixes) as surface-level features. It also reports that adding head and tail characters, a list of sukun words, and named entities further improved results.
- **Gold:** affixes, leading and trailing characters in words and stems, and the presence of words in large gazetteers of named entities
- **Retrieved:** ['p00037', 'p00034', 'p00033', 'p00041', 'p00050']; **cited:** ['p00034', 'p00041']; **gold evidence (best of 1 references):** ['p00019']
- **Missed** `p00019` [Our Diacritizer ::: Case Ending Diacritization ::: Features.]: Table TABREF17 lists the features that we used for CE recovery. We used Farasa to perform segmentation and POS tagging and to determine stem-templates BIBREF31. Farasa has a reported POS accuracy of 96% on the WikiNews dataset BIBREF31. Tho

### Q13 `1907.09369`: retrieval_miss

- **Question:** What baseline is used?
- **Prediction:** The baseline uses a bag-of-words (BOW) model with a maximum entropy classifier (paper 1907.09369).
- **Gold:**  Wang et al. BIBREF21, paper by BIBREF33 in which they used maximum entropy classifier with bag of words model to classify various emotional datasets
- **Gold:** Wang et al. , maximum entropy classifier with bag of words model
- **Retrieved:** ['p00031', 'p00036', 'p00029', 'p00013', 'p00014']; **cited:** ['p00029']; **gold evidence (best of 2 references):** ['p00009', 'p00010', 'p00011']
- **Missed** `p00009` [Baseline Approaches]: We compare our approach to two other, the first one uses almost the same tweet data as we use for training, and the second one is the CrowdFlower dataset annotated for emotions.
- **Missed** `p00010` [Baseline Approaches]: In the first one Wang et al. BIBREF21 downloaded over 5M tweets which included one of 131 emotional hashtags based on Parrott's three-level categorization of emotions in seven categories: joy, sadness, anger, love, fear, thankfulness, surpr
- **Missed** `p00011` [Baseline Approaches]: In the second one, the reported results are from a paper by BIBREF33 in which they used maximum entropy classifier with bag of words model to classify various emotional datasets. Here we only report part of their result for CrowdFlower data

### Q14 `1908.04042`: retrieved_answered

- **Question:** how is diversity measured?
- **Prediction:** In paper 1908.04042, diversity is measured as the average dissimilarity between all pairs of tags in the recommended tag list.
- **Gold:** average dissimilarity of all pairs of tags in the list of recommended tags
- **Gold:**  the average dissimilarity of all pairs of tags in the list of recommended tags
- **Retrieved:** ['p00027', 'p00035', 'p00051', 'p00006', 'p00060']; **cited:** ['p00035']; **gold evidence (best of 2 references):** ['p00035']

### Q15 `1909.09067`: retrieval_partial

- **Question:** Which information about text structure is included in the corpus?
- **Prediction:** The corpus includes text-structure information such as paragraphs and lines (paper 1909.09067).
- **Gold:** paragraphs, lines, Information on physical page segmentation (for PDFs only), paragraph segmentation, and line segmentation
- **Gold:** paragraph, lines, textspan element (paragraph segmentation, line segmentation, Information on physical page segmentation(for PDF only))
- **Retrieved:** ['p00013', 'p00035', 'p00047', 'p00044', 'p00006']; **cited:** ['p00006']; **gold evidence (best of 2 references):** ['p00006', 'p00037']
- **Missed** `p00037` [Building a Corpus for Automatic Readability Assessment and Automatic Text Simplification of German ::: Secondary Data]: Information on physical page segmentation (for PDFs only), paragraph segmentation, and line segmentation was added as part of a textspan element in the textstructure layer

### Q16 `1911.07555`: over_abstain

- **Question:** Is the lexicon the same for all languages?
- **Prediction:** Unanswerable
- **Gold:** Yes
- **Gold:** Yes
- **Retrieved:** ['p00028', 'p00015', 'p00018', 'p00019', 'p00006']; **cited:** []; **gold evidence (best of 2 references):** ['p00018']

### Q17 `1705.04153`: missed_abstain

- **Question:** What is the meta knowledge specifically?
- **Prediction:** In paper 1705.04153, the meta knowledge is task-specific semantic information for simple tasks like text classification, and syntactic information for more complex tasks like semantic matching. One example is recognizing light-verb phrases such as “taking off” for a forthcoming compositional operation.
- **Gold:** Unanswerable
- **Retrieved:** ['p00071', 'p00073', 'p00054', 'p00046', 'p00075']; **cited:** ['p00071', 'p00073']; **gold evidence (best of 1 references):** []

### Q18 `2001.03131`: missed_abstain

- **Question:** How do their results compare to state-of-the-art?
- **Prediction:** The paper does not provide the proposed RKS results or a direct comparison with state-of-the-art scores in the supplied passages. It reports that the top SemEval 2019 Sub-task A team achieved an F1-score of 82.9%.
- **Gold:** Unanswerable
- **Gold:** Unanswerable
- **Retrieved:** ['p00026', 'p00039', 'p00008', 'p00007', 'p00030']; **cited:** ['p00026', 'p00030']; **gold evidence (best of 2 references):** []

### Q19 `1911.07555`: missed_abstain

- **Question:** What are the languages represented in the DSL datasets? 
- **Prediction:** The DSL 2017 dataset represents 14 languages across 6 language groups (the passage does not name them).
- **Gold:** Unanswerable
- **Retrieved:** ['p00009', 'p00012', 'p00004', 'p00028', 'p00001']; **cited:** ['p00009']; **gold evidence (best of 1 references):** []
