#https://www.youtube.com/watch?v=sa-TUpSx1JA
r"""These are several sentences. They will be splittet a lot. 
It is inevitable. It will happen although J. D. Gould
would like it to be otherwise, or se he says. 
This sentence tests (or intends to) test parenthes
and explamations! At least that was the plan.
Another thing one might do is the following: testing this."""

#one funny thing
321-123-412-213
142.125.123.125

#another:
cat 
mat 
bat 
hat

#another: M(r|s|rs)\.?\s[A-Z]\w*
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T 

#last 
CoreyMSchafer@gmail.com
corey.schafer@university.edu 
corey-321-schafer@my-work.net 

#capturing 
https://www.google.com 
http://coreyms.com 
https://youtube.com 
https://www.nasa.gov 

#https://www.youtube.com/watch?v=K8L6KVGG-7o
import re

search_txt = '''
These are several sentences. They will be splittet a lot. 
It is inevitable. It will happen although J. D. Gould
would like it to be otherwise, or se he says. 
This sentence tests (or intends to) test parenthes
and explamations! At least that was the plan.
Another thing one might do is the following: testing this.
abc, and abc. 
'''

pattern = re.compile(r'abc')
matches = pattern.finditer(search_txt)

for match in matches: 
    print(match) #returning index. 

print(search_txt[317:320]) #and there it is. 

help(re.compile)