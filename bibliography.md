
# Table of Contents

1.  [Summarization](#org9d4f43b)
    1.  [Extraction](#orgd1748e5)
        1.  [Building an Extraction](#orge137432)
    2.  [Abstraction](#org75ff3b1)
2.  [References](#orgfb629e1)
    1.  [Brief Introduction to NLP](#orgda71e4a)
    2.  [Overview of Text Summarization Techniques](#org84dfffc)
        1.  [See Section 5 for further references to review for conversation summaries](#orgc6d85c7)
        2.  [Nathan: See section 7](#org7fac88f)


<a id="org9d4f43b"></a>

# Summarization


<a id="orgd1748e5"></a>

## Extraction

Identifies important pieces of text within a corpus (body of text) and builds a
summary which contains only those words.


<a id="orge137432"></a>

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
    
    1.  Frequency Approach
    
        1.  Word Probability
        
            Probability of a word occuring in a document.
            
            \begin{eq1}
                P(w) = \frac{f(w)}{N}
            \end{eq1}
            
            For each sentence, the average probability of a word is assigned as a *weight*.
            Then, the best scoring sentence with the highest probability word is chosen to
            ensure that the sentence is present in the summary. The weight of the chosen
            word is then updated to ensure that a word in the summary is not chosen over a
            word that only occurs once<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>.
            
            \begin{eq2}
                p_{new}(w_i) = p_{old}(w_i) p_{old}(w_i)
            \end{eq2}
        
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


<a id="org75ff3b1"></a>

## Abstraction

Interprets and analyzes important pieces of text within a corpus and builds a
human readable summary. This is more advanced and computation-intenseive than
Extraction.


<a id="orgfb629e1"></a>

# References


<a id="orgda71e4a"></a>

## [Brief Introduction to NLP](https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f)


<a id="org84dfffc"></a>

## [Overview of Text Summarization Techniques](https://arxiv.org/pdf/1707.02268.pdf)


<a id="orgc6d85c7"></a>

### See Section 5 for further references to review for conversation summaries


<a id="org7fac88f"></a>

### Nathan: See section 7


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> Unsure of this in particular. Need confirmation
