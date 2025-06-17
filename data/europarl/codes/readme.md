dataset downloaded from:
https://www.statmt.org/europarl/

Preprocessing codes

first we select several languages and bring them to a new directory.
we select cs/en/es/it/nl/pl/ro/sl

We first run a script to only keep the files that are shared between all the languages.

for example if a file like `ep-10-12-16-019.txt` is shared between all languages but is not in `sl` we will remove it from all langues.

then we remove files that are smaller than 2kB.

then we try to remove very short sentences. we remove the sentences that are shorter than x in their english version.
we remove the files that are misaligned in the number of lines.
we remove the meta lines. lines that start with `<`.
we merge all files in each directory to have single file called processed.txt
we make a new file called long.txt in each directory which is the file that is longer than 100 tokens in English version.


Link to my conversation about this with ChatGPT:

https://chatgpt.com/share/67e2e533-9a34-8009-befd-5f263ecf92f0
