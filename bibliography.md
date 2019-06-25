
# Table of Contents

1.  [Summarization](#org1f671d9)
    1.  [Extraction](#orge403ab5)
        1.  [Building an Extraction](#org67678e8)
    2.  [Abstraction](#org18dbf79)
    3.  [Evaluating Summaries](#orgd61f884)
        1.  [Human Evaluation](#orgbb4c049)
        2.  [Recall-Oriented Understudy for Gisting Evaluation (ROUGE)](#org9e3d9cf)
2.  [References](#org77a030b)
    1.  [Brief Introduction to NLP](#org920df0c)
    2.  [Overview of Text Summarization Techniques](#orgf0bae6f)
        1.  [See Section 5 for further references to review for conversation summaries](#org51b6963)
        2.  [Nathan: See section 7](#org31fd8b0)


<a id="org1f671d9"></a>

# Summarization


<a id="orge403ab5"></a>

## Extraction

Identifies important pieces of text within a corpus (body of text) and builds a
summary which contains only those words.


<a id="org67678e8"></a>

### Building an Extraction

**Steps**

1.  Construct an IR (Intermediate Representation)
2.  Score Sentences based on a scoring algorithm
3.  Select a summary based on the scored sentences

1.  IR

    1.  Topic Representation
    
        Interprets the topics discussed in the corpus. May use a Frequency approach, sentiment analysis, topic word (dictionary), or Bayesian
        Topic Model approach.
        
        1.  Frequency
        
            Frequency of words used to determine a *topic*. This can be taken a step
            further by using the Log-Likelihood Ratio Test.
    
    2.  Indicator Representation
    
        Describes every sentence as a list with important covariates such as word count, length, position in the document, and presence
          of keywords.

2.  Sentence Score

    In a topic IR, this score is an indicator of importance of the sentence. In an
    Indicator IR, this score is some model based off the covariates. Importance of a sentence can be determined
    either by **count** or **proportion** of topic words.

3.  Summary Selection

    Selects the *k* most important sentences for the summary. Additional criteria
    beyond the score may be assessed to determine the sentences chosen for the
    summary. i.e. Type of document (Newspaper, Blog, Magazine, etc.)
    
    1.  Topic Representation
    
        1.  Frequency Approach
        
            1.  Word Probability
            
                Probability of a word occuring in a document.
                
                \begin{math}
                    P(w) = \frac{f(w)}{N}
                \end{math}
                
                For each sentence, the average probability of a word is assigned as a *weight*.
                Then, the best scoring sentence with the highest probability word is chosen to
                ensure that the sentence is present in the summary. The weight of the chosen
                word is then updated to ensure that a word in the summary is not chosen over a
                word that only occurs once <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>.
                
                \begin{math}
                    p_{new}(w_i) = p_{old}(w_i) p_{old}(w_i)
                \end{math}
            
            2.  Term Frequency Inverse Document Frequency (TFIDF)
            
                A weighting technique which penalizes words that occur most frequently in a
                document.
                
                \begin{eq3}
                    q(w) = f_d(w) \log(\frac{|D|}{f_D(w)})
                \end{eq3}
                
                \(f_d(w)\): Term frequency of a word (w) in a document (d)
                \(f_D(w)\): Number of documents that contain the word (w)
                \(|D|\): Number of documents in a collection (D)
                
                -   easy and fast to compute.
                -   Used in many text summarizers
                
                1.  Centroid-based Summarization
                
                    A method of ranking sentences based on TFIDF
                    **Steps**
                    
                    1.  Detect Topics and documents that describe the same topic clustered together
                        -   TFIDF vectors are calculated and TFIDF scores below a predefined threshold
                            are removed
                    2.  Clustering Algorithm is run over TFIDF vectors and centroids (median of a
                        cluster) are recomputed after each document is added.
                        -   Centroids may be considered pseudo-documents which contain a higher than
                            the predefined TFIDF threshold.
                    3.  Use Centroids to find sentences related to the topic central to the cluster
                        -   Cluster-based Relative Utility (CBRU) describes how relevant the topic is
                            to the general topic of the cluster.
                        -   Cross Sentence Informational Subsumption (CSIS) measures redundancy between
                            sentences
        
        2.  Latent Semantic Analysis
        
            Unsupervised method to selected highly ranked sentences for single and
            multi-document summaries. Let an *n x m* matrix exist where \(n_i\) is a word in
            the corpus and \(m_j\) is a sentence. Each entry \(a_{ij}\) is the TFIDF weight for
            given word and sentence. Singular Value Decomposition (SVD) is then applied to
            retrieve three matrices: \(A = U \Sigma V^T\) where \(D = \Sigma V^T\) describes the
            relationship between a sentence and a topic.
            
            The assumption is that a topic can be expressed in a single sentence which is
            not always the case. Additional alternatives have been suggested to overcome
            this assumption.
        
        3.  Bayesian Topic Models
        
            Using probability distributions to model probability of words overcomes two
            limitations present in other methods:
            
            1.  Sentences are assumed to be independent so topics embedded in documents are
                ignored
            2.  Sentence scores are heuristics and therefore hard to interpret
            
            The scoring used in Bayesian topic models is typically the Kullbak-Liebler (KL)
            which measures the difference between two probability distributions P and Q.
    
    2.  Indicator Representation
    
        1.  Graph
        
            Represent documents as a graph. Often influenced by PageRank. Sentences are the
            vertices and edges are similarity (weights). Most common weight is cosine
            similarity against TFIDF weights for given words.
        
        2.  Machine Learning
        
            Approach summarization as a classification problem. Machine Learning techniques include:
            
            -   Naive Bayes
            -   Decision Trees
            -   Support Vector Machines
            -   Hidden Markov Models\*
            -   Conditional Random Fields\*
                \*Assume Dependence
                
                Models that assume dependence often outperform those who do not.


<a id="org18dbf79"></a>

## Abstraction

Interprets and analyzes important pieces of text within a corpus and builds a
human readable summary. This is more advanced and computation-intenseive than
Extraction.


<a id="orgd61f884"></a>

## Evaluating Summaries

Principles in evaluating whether a summary is good or not

1.  Decide and specify the most important parts of the original text
2.  Identify important info in the candidate summary since the information can be
    represented using disparate expressions.
3.  Readability


<a id="orgbb4c049"></a>

### Human Evaluation

Self explanatory.


<a id="org9e3d9cf"></a>

### Recall-Oriented Understudy for Gisting Evaluation (ROUGE)

Determine the quality of a summary by comparing it to human summaries.

1.  ROUGE-n

    **gram**: a word
    A series of n-grams is created from the reference summary and the candidate
    summary (usually 2-3 and rarely 4 grams).
    \(p\) = number of common n-grams
    \(q\) = number of n-grams from reference summary
    \[
    {ROUGE-n} = \frac{p}{q}
    \]

2.  ROUGE-l

    Longest Common Subsequence (LCS) betweeen two sequences of text. The longer the
    LCS, the more similar they are. Requires ordering to be the same.

3.  ROUGE-SU

    Also called *skip-bi-gram* and *uni-gram*.
    Allows insertion of words between the first and last words of bi-grams so
    consecutive words are not needed unlike ROUGE-n and ROUGE-l.


<a id="org77a030b"></a>

# References


<a id="org920df0c"></a>

## [Brief Introduction to NLP](https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f)


<a id="orgf0bae6f"></a>

## [Overview of Text Summarization Techniques](https://arxiv.org/pdf/1707.02268.pdf)


<a id="org51b6963"></a>

### See Section 5 for further references to review for conversation summaries


<a id="org31fd8b0"></a>

### Nathan: See section 7


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> Unsure of this in particular. Need confirmation
