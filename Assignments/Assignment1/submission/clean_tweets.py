import re

def clean(s):
    """
    A function to clean the input string
    It ensures that there is at least one
    space between intended tokens; then
    one can tokenise easily by splitting
    along spaces.
    """
    # The following lines clean the text by inserting placeholders
    # and removing repetition.

    s = s.lower()
    # Convert to lowercase
    s = re.sub(r'#[a-zA-Z\d]+'                   ,  '<HASHTAG>', s)
    # Hashtags
    s = re.sub(r'[a-zA-Z\.]+@[a-zA-Z\.\d]+'      ,  '<EMAIL>'  , s)
    # Email IDs
    s = re.sub(r'@[a-zA-Z\.\d_]+'                ,  '<MENTION>', s)
    # Mentions
    s = re.sub(r'\d+(,(\d+))*(\.(\d+))?%?\s'     ,  '<NUMBER> ', s)
    # Numbers: digits, optionally followed by a comma and more digits,
    #          optionally followed by a decimal point and more digits,
    #          optionally followed by a percentage symbol
    s = re.sub(r'[a-zA-Z:/\d]+\.[a-zA-Z/\d]+(\.[a-zA-Z/\d]+)*(/[a-zA-Z/\d=&\?~]+)*',
                                                    '<URL>'    , s)
    # URLs: a sequence of letters, colons, slashes; a period;
    #       a sequence of letters, slashes, digits;
    #       optionally more periods and sequences;
    #       optionally more slashes and sequences
    s = re.sub(r'\d\d:\d\d\s?(([AP]M)|([ap]m))'  ,  '<TIME>'   , s)
    # Time expressions of the form HH:MM AM/PM/am/pm
    s = re.sub(r'\$\d+(,\d+)*(\.\d+)?%?'         ,  '<MONEY>'  , s)
    # Money expressions in dollars
    s = re.sub(r'([,\.!\?<>\-/\(\)])\1+'         , r'\1'       , s)
    # Repeated punctuation

    # The following line replaces all strings of more than one space/tab
    # with a single space, and cuts out leading and trailing spaces
    s = re.sub(r'\s+'                            , ' '         , s)
    s = re.sub(r'^\s+'                           , ''          , s)
    s = re.sub(r'\s+$'                           , ''          , s)

    return s

def tokenize(s):
    """
    Tokenizes by cleaning the string,
    separating punctuation and
    placeholders, and
    splitting along spaces
    """
    s = clean(s)

    s = re.sub(r'([,\.!\?\-;:"&\+\(\)/\[\]])'    , r' \1 '      , s)
    # Separating out punctuation
    s = re.sub(r'(\'(s|m|re|ll|d|ve))\s'         , r' \1 '      , s)
    # Contractions and genitive markers are separate tokens
    s = re.sub(r'([^(ca)(wo)])(n\'t)\s'          , r'\1 \2 '    , s)
    # `n't` is a separate token, except in the cases of `can't` and `won't`
    s = re.sub( 'cannot'                         ,  'can not'   , s)
    # "cannot" is "can not"
    s = re.sub(r'can\'t\s'                       ,  "can n't "  , s)
    s = re.sub(r'won\'t\s'                       ,  "will n't " , s)
    s = re.sub(r'shan\'t\s'                      ,  "shall n't ", s)
    # `can't` is `can n't` and `won't` is `will n't`

    s = re.sub(r'(<((HASHTAG)|(EMAIL)|(MENTION)|(URL)|(TIME)|(MONEY)|(NUMBER))>)',
                                                   r' \1 '     , s)
    # Separating placeholders

    tokens = s.split()
    return tokens

# Driver
tweets_corpus = open("../corpora/general-tweets.txt", 'r')
cleaned_corpus = open("2020114001_tokenize.txt", 'w')

for tweet in tweets_corpus:
    cleaned_corpus.write(clean(tweet) + '\n')

tweets_corpus.close()
cleaned_corpus.close()
