Codes:
1. sameordergeneratedata.py - Data Generating Code
2. evaluate.py - Code to collect the output of the model to test its original performance
3. collectactivation.py - Code to collect activations for needed for probing
4. probewcontrol27.py - Code to probe token 27
5. probewcontrolsecondLast.py - Code to probe second last token
6. patchtokens.py - Code to perform intervention at most token to check arithmetic composition
7. patchhead.py and valwattn.py - The first code is needed to collect results of intervention on individual heads and the second code is needed to collect value-weighted attention. Together, the results of these codes are needed to compute the token importance score
8. KeysSwap.py - Code needed to perform interventions to check what the model prioritizes for retrieval and composition.
   
Packages needed for intervention could be found here:
1. https://nnsight.net/start/ (for all except Key Swap)
2. https://colab.research.google.com/drive/1wjQhbQKh2pwy-mxx4EFMBC1IEauzu9G0#scrollTo=jAGBcXiMrzFg (for key swap use this version)

   
