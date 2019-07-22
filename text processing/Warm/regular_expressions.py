import sys, re

sent = "I have baked a Cake!\n"

# extracting matched parts
# match = re.search(' (([a-z]+)(ed|ing))', sent)
# match.group()
# match.span()

# findbhjing multiple regex matches
# multi_match = re.compile('[A-Za-z\n]+')
# multi_match.findall(sent)

re.split(r'(\W+)', sent)
