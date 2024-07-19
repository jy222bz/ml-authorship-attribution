#  Stylometry & Machine Learning: Authorship Attribution


## XGBoost and Stylometric Analysis Of Classic Literature

## Project Overview

The study delves into classic literature utilizing various techniques, including Natural Language Processing (NLP), Machine Learning (ML),  and Stylometric Analysis. Through this multidimensional approach, it unveils the features of the texts and the authors.

## Objectives

- To utilize NLP and ML methodologies to extract and examine stylistic elements within classic texts.
- To conduct a stylometric analysis on the literary styles of the 19th century.
- To conduct authorship attribution using stylometry analysis.

## Getting Started

### Prerequisites

Ensure that Python 3.8 or later is installed on the system. This project relies on several Python libraries, including but not limited to:

- `gensim`
- `imbalanced-learn`
- `matplotlib`
- `nltk`
- `numpy`
- `optuna`
- `pandas`
- `plotly`
- `scipy`
- `seaborn`
- `spacy`
- `sklearn`
- `textatistic`
- `textblob`
- `textstat`
- `torch`

### Installing Dependencies:
```bash
pip install -r requirements
```

### For Anaconda users
Ensure Anaconda is installed on the system and then import the `env.yaml`. If not, download and install it from the [official Anaconda website](https://www.anaconda.com/).

Clone the project repository:

```bash
git clone https://github.com/jy222bz/ml-authorship-attribution.git
```

## Subject Literature 
### The total number of literature pieces is 108, with each of the 18 unique authors - making it six books per author, resulting in 108 unique books.

| Book Name                                                   | Author             |
|-------------------------------------------------------------|--------------------|
| A Christmas Carol In Prose                                  | Charles Dickens    |
| A Connecticut Yankee in King Arthur's Court                 | Mark Twain         |
| A Journal Of Impressions In Belgium                         | May Sinclair       |
| A Little Country Girl                                       | Susan Coolidge     |
| Adam Bede                                                   | George Eliot       |
| Adventures Of Huckleberry Finn                              | Mark Twain         |
| All Cats Are Gray                                           | Andre Norton       |
| Anne Of Avonlea                                             | L. M. Montgomery   |
| Anne Of Green Gables                                        | L. M. Montgomery   |
| Anne Of The Island                                          | L. M. Montgomery   |
| Argonaut Stories                                            | Jack London        |
| A Round Dozen                                               | Susan Coolidge     |
| Bleak House                                                 | Charles Dickens    |
| Clover                                                      | Susan Coolidge     |
| Curious-If True                                             | Elizabeth Gaskell  |
| Daniel Deronda                                              | George Eliot       |
| David Copperfield                                           | Charles Dickens    |
| Defense Mech                                                | Ray Bradbury       |
| Eight Cousins                                               | Louisa May Alcott  |
| Emily Of New Moon                                           | L. M. Montgomery   |
| Emma                                                        | Jane Austen        |
| Ethan Frome                                                 | Edith Wharton      |
| Felix Holt-The Radical                                      | George Eliot       |
| Flower Fables                                               | Louisa May Alcott  |
| Futuria Fantasia-Fall                                       | Ray Bradbury       |
| Futuria Fantasia-Spring                                     | Ray Bradbury       |
| Great Expectations                                          | Charles Dickens    |
| Half a Life-Time Ago                                        | Elizabeth Gaskell  |
| Hard Times                                                  | Charles Dickens    |
| Heart Of Darkness                                           | Joseph Conrad      |
| In The High Valley                                          | Susan Coolidge     |
| Jack And Jill                                               | Louisa May Alcott  |
| Key Out Of Time                                             | Andre Norton       |
| Let's Get Together                                          | Isaac Asimov       |
| Life And Death Of Harriett Frean                            | May Sinclair       |
| Life On The Mississippi                                     | Mark Twain         |
| Little Women                                                | Louisa May Alcott  |
| Lord Jim                                                    | Joseph Conrad      |
| Mansfield Park                                              | Jane Austen        |
| Martin Eden                                                 | Jack London        |
| Mary Barton                                                 | Elizabeth Gaskell  |
| Ninety-Three                                                | Victor Hugo        |
| Northanger Abbey                                            | Jane Austen        |
| Notre-Dame de Paris                                         | Victor Hugo        |
| Oliver Twist                                                | Charles Dickens    |
| Persuasion                                                  | Jane Austen        |
| Pillar Of Fire                                              | Ray Bradbury       |
| Plague Ship                                                 | Andre Norton       |
| Poirot Investigates                                         | Agatha Christie    |
| Pride And Prejudice                                         | Jane Austen        |
| Rainbow Valley                                              | L. M. Montgomery   |
| Rilla Of Ingleside                                          | L. M. Montgomery   |
| Rose In Bloom                                               | Louisa May Alcott  |
| Roughing It                                                 | Mark Twain         |
| Ruth                                                        | Elizabeth Gaskell  |
| Sense And Sensibility                                       | Jane Austen        |
| Silas Marner                                                | George Eliot       |
| Star Born                                                   | Andre Norton       |
| Summer                                                      | Edith Wharton      |
| Tales Of Men and Ghosts                                     | Edith Wharton      |
| The Ancient Law                                             | Ellen Glasgow      |
| The Battle Ground                                           | Ellen Glasgow      |
| The Builders                                                | Ellen Glasgow      |
| The Combined Maze                                           | May Sinclair       |
| The Creatures That Time Forgot                              | Ray Bradbury       |
| The Custom Of The Country                                   | Edith Wharton      |
| The Genetic Effects Of Radiation                            | Isaac Asimov       |
| The Grey Woman And Other Tales                              | Elizabeth Gaskell  |
| The History Of A Crime                                      | Victor Hugo        |
| The House Of Mirth                                          | Edith Wharton      |
| The Innocents Abroad                                        | Mark Twain         |
| The Iron Heel                                               | Jack London        |
| The Lifted Veil                                             | George Eliot       |
| The Man In The Brown Suit                                   | Agatha Christie    |
| The Man Who Laughs-A Romance Of English History             | Victor Hugo        |
| The Mill On The Floss                                       | George Eliot       |
| The Murder On The Links                                     | Agatha Christie    |
| The Mysterious Affair At Styles                             | Agatha Christie    |
| The Mystery Of The Blue Train                               | Agatha Christie    |
| The People Of The Abyss                                     | Jack London        |
| The People Of The Crater                                    | Andre Norton       |
| The Prince And The Pauper                                   | Mark Twain         |
| The Reef                                                    | Edith Wharton      |
| The Romance Of A Plain Man                                  | Ellen Glasgow      |
| The Sea-Wolf                                                | Jack London        |
| The Secret Adversary                                        | Agatha Christie    |
| The Secret Agent-A Simple Tale                              | Joseph Conrad      |
| The Secret Sharer                                           | Joseph Conrad      |
| The Shape Of Things                                         | Ray Bradbury       |
| The Three Sisters                                           | May Sinclair       |
| The Time Traders                                            | Andre Norton       |
| The Tree Of Heaven                                          | May Sinclair       |
| The Voice Of The People                                     | Ellen Glasgow      |
| The Women Who Make Our Novels                               | Ellen Glasgow      |
| Toilers Of The Sea                                          | Victor Hugo        |
| Uncanny Stories                                             | May Sinclair       |
| Under The Lilacs                                            | Louisa May Alcott  |
| Under Western Eyes                                          | Joseph Conrad      |
| Victory-An Island Tale                                      | Joseph Conrad      |
| What Katy Did at School                                     | Susan Coolidge     |
| What Katy Did Next                                          | Susan Coolidge     |
| White Fang                                                  | Jack London        |
| William Shakespeare                                         | Victor Hugo        |
| Wives And Daughters                                         | Elizabeth Gaskell  |
| Worlds Within Worlds-The Story Of Nuclear Energy-Vol1       | Isaac Asimov       |
| Worlds Within Worlds-The Story Of Nuclear Energy-Vol2       | Isaac Asimov       |
| Worlds Within Worlds-The Story Of Nuclear Energy-Vol3       | Isaac Asimov       |
| Youth                                                       | Isaac Asimov       |

### Disclaimer

The literature contained in these files has been retrieved from Project Gutenberg. All content specifically related to Project Gutenberg, such as disclaimers, licensing information, and other project-related content, has been removed to ensure the files contain only the actual contents of the books. The source of the books remains Project Gutenberg, and the literary content of the books has not been altered in any way. This process was done solely to facilitate the preparation of the book contents for processing and analysis without unrelated content.

For more information on Project Gutenberg, please visit [Project Gutenberg's website](https://www.gutenberg.org).


## Instructions for Running the Solution
To run the solution, follow these steps:

1. Clone the repository and install the necessary environment requirements.
2. Review the solution, current results, and plots to understand the computations, analysis, and learning process of authorship attribution and stylistic analysis based on stylometric methods.
3. To rerun the solution using the same or different configurations, empty all directories except the `Literature` directory and the `books.json` file in the `Datasets` directory. These must remain intact unless you wish to use different books and configurations, in which case you should update the configurations accordingly.
4. Create the necessary folders required under the root directory for the solution, such as the `Model` directory, which will be used to extract the trained model.
5. Run the notebooks sequentially in the order they are named.


## Author

- **Jacob Yousif** 
  
