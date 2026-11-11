import urllib.request

inputList = ['INCHI', 'INCHIKEY', 'MolecularFormula', 'CanonicalSMILES', 'MolecularWeight']
selChoice = 'INCHI'
#%%
string1 = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
string3 = "/property/"
string4 = "/TXT"
#%%
odor_name = 'isoamyl acetate'
string2 = odor_name
html = urllib.request.urlopen(string1 + string2 + string3 + selChoice + string4).read()
html2 = html.decode('UTF-8')
#%%

from pathlib import Path

inchi_str = "1S/C5H4O2/c6-4-5-2-1-3-7-5/h1-4H"
inchi = f"InChI={inchi_str}"

prolog = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
id_input = f"compound/inchi/{inchi}"
output = 'synonyms/TXT'

url = Path(prolog) / Path(id_input) / Path(output)
print(url)

"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/cids/txt?inchi={Inchi}}"
#%%
"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/cids/txt?inchi=InCC1%3DC%28%5BC%40%40%5D%28SC1%3DO%29%28C" \
"%29" \
"%2FC%3DC%28%5CC%29%2FC%3DC%29O"