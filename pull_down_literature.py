import os
import ast 
import sys
import json
import tarfile
import xmltodict
import requests
import urllib.request
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from itertools import chain
from lxml import etree
from Bio import Entrez
from Bio import Medline
from bs4 import BeautifulSoup

def evaluate(term_list, literature_dir, solution):
    """
    """
    solution = [x.lower() for x in solution]
    print(solution)
    free_text_files = [os.path.join(literature_dir, x) for x in os.listdir(literature_dir) if x.endswith("free_text.txt")]
    print(free_text_files)
    for filename in free_text_files:
        with open(filename, 'r') as rfile:
            for line in rfile:
                text_body = line.strip()
        text_body = text_body.lower()

        #check and see which of the solution steps are included in this literature search
        for solution_step in solution: 
            if solution_step in text_body:
                print(filename, solution_step)
    
        #sys.exit(0)
    sys.exit(0)
def untar(literature_dir):
    tar_dir = os.path.join(literature_dir, "tar")
    all_tar_files = [os.path.join(tar_dir, x) for x in  os.listdir(tar_dir)]
    for filename in all_tar_files:
        base_name = filename.replace(".tar.gz","")
        if os.path.isdir(base_name):
            continue
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(path=tar_dir)
        tar.close()

def count_pmc(term_list, literature_dir):
    counts = []
    for term in term_list:
        pmc_count = 0
        term_filename = os.path.join(literature_dir, "{term}.txt".format(term=term.replace(" ","_")))
        if not os.path.isfile(term_filename):
            continue
        try:
            df = pd.read_table(term_filename, header=None)
            pmc_count += len(df)
        except:
            pass        
        counts.append(pmc_count)
    return(counts)

def ftp_fetch_articles(term_list, literature_dir):
    """
    For a list of terms, file the text file with PMC ids and use those PMC ids to FTP fetch the article and write it to a text file.
    """
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id="
    for term in term_list:
        term_filename = os.path.join(literature_dir, "{term}.txt".format(term=term.replace(" ","_")))
        if not os.path.isfile(term_filename):
            continue
        try:
            df = pd.read_table(term_filename, header=None)
        except:
            print("error parsing file", term_filename)
            continue
        pmc_ids = df[0].tolist()
        for i,pmc in enumerate(pmc_ids):
            if i % 10 == 0 and i != 0:
                print("Percent completed %s FTP download:" %term, i/len(pmc_ids))
            output_loc = './text_data/tar/%s.tar.gz' %pmc
            if os.path.isfile(output_loc):
                continue
            new_url = base_url + pmc
            resp = requests.get(new_url)
            resp_dict = xmltodict.parse(resp.text)
            if 'error' in resp_dict['OA']:
                print(resp_dict)
                print("error has occured", pmc)
                continue
            #print(pmc, resp_dict)
            get_link = resp_dict['OA']['records']['record']['link']
            if type(get_link) == list:
                for link in get_link:
                    if link['@format'] == 'tgz':
                        ftp_loc = link['@href']
                        urllib.request.urlretrieve(ftp_loc, output_loc)
            elif type(get_link) == dict:
                ftp_loc = get_link['@href']
                urllib.request.urlretrieve(ftp_loc, output_loc)

def stringify_children(node):
    """
    Filters and removes possible Nones in texts and tails
    ref: http://stackoverflow.com/questions/4624062/get-all-text-inside-a-tag-in-lxml
    """
    parts = (
        [node.text]
        + list(chain(*([c.text, c.tail] for c in node.getchildren())))
        + [node.tail]
    )
    parts = "".join(filter(None, parts))
    parts = parts.strip()
    #print(parts)
    return(parts)

def parse_free_text(term_list, literature_dir):
    """
    Given a list of entities, open text files in medline format containing relevant information and parse into something useable in downstream processing.
    """
    tar_dir = os.path.join(literature_dir, "tar")
    all_subdirs = [os.path.join(tar_dir, x) for x in os.listdir(tar_dir)]
    all_subdirs = [x for x in all_subdirs if os.path.isdir(x)]
    abbreviation_dict = {}
    for i, subdir in enumerate(all_subdirs):
        pmid = subdir.split("/")[-1]
        free_text_filename = os.path.join(literature_dir, "%s_free_text.txt" %pmid)
        print(i/len(all_subdirs))
        all_files = [os.path.join(subdir, x) for x in os.listdir(subdir)]
        xml_files = [x for x in all_files if x.endswith("xml")]
        if len(xml_files) > 1:
            print("multiple xml files")
            sys.exit(1)
        file_path = xml_files[0]
        tree = read_xml(file_path)
        paragraphs = tree.xpath("//body//p")
        tmp_text = ""
        for paragraph in paragraphs:
            paragraph_text = stringify_children(paragraph).replace("\n"," ")
            tmp_text += paragraph_text.strip() + " "
        with open(free_text_filename, 'w') as ffile:
            ffile.write(tmp_text)

def chunk_text(free_text, size):
    text_list = []
    sentences = free_text.split(".")
    count = [len(x.split(" ")) for x in sentences]
    tmp = ""
    tmp_count = 0
    for c, s in zip(count, sentences):
        if tmp_count > size:
            text_list.append(tmp)
            tmp = ""
            tmp_count = 0
        else:
            tmp_count += c
            tmp += s + '. '
    return(text_list)

def read_xml(path, nxml=True):
    """
    Parse tree from given XML path
    """
    try:
        tree = etree.parse(path)
        if ".nxml" in path or nxml:
            remove_namespace(tree)  # strip namespace when reading an XML file
    except:
        try:
            tree = etree.fromstring(path)
        except Exception:
            print(
                "Error: it was not able to read a path, a file-like object, or a string as an XML"
            )
            raise
    return tree

def remove_namespace(tree):
    """
    Strip namespace from parsed XML
    """
    for node in tree.iter():
        try:
            has_namespace = node.tag.startswith("{")
        except AttributeError:
            continue  # node.tag is not a string (node is a comment or similar)
        if has_namespace:
            node.tag = node.tag.split("}", 1)[1]


def parallel_fetch_pmc_ids(term_list, literature_dir):
    results = Parallel(n_jobs=3, verbose=0)(delayed(fetch_pmc_ids)(term, literature_dir) for term in term_list)

def parallel_fetch_abstract_counts(term_list):
    results = Parallel(n_jobs=1, verbose=0)(delayed(fetch_pubmed_abstract_counts)(term) for term in term_list)
    return(results)

def fetch_pubmed_abstract_counts(term):
    """
    Given a list of terms, fetch PMC ids for free articles associated with the search terms and write it to a text file.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    Entrez.email = "caceves@scripps.edu"
    num_abstracts = 1000
    handle = Entrez.esearch(db="pubmed", term=term + " AND fha[Filter]", retmode="xml", retmax=num_abstracts, sort="relevance", usehistory='n')
    records = Entrez.read(handle)
    pmids = records['IdList']
    webenv = ""
    query_key = ""
    pmids_url = ""
    return(len(pmids))    



def fetch_pubmed_abstracts(term, literature_dir, num_abstracts, terms):
    """
    Given a list of terms, fetch PMC ids for free articles associated with the search terms and write it to a text file.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    Entrez.email = "caceves@scripps.edu"

    term_filename = os.path.join(literature_dir, "{term}.json".format(term=term.replace(" ","_").replace("/","").replace("(", "").replace(")","")))
    if os.path.isfile(term_filename):
        return(0)

    out_handle = open(term_filename, "w")
    handle = Entrez.esearch(db="pubmed", term=term + " AND fha[Filter] AND review[Filter] AND humans[Filter]", retmode="xml", retmax=num_abstracts, sort="relevance", usehistory='n')
    records = Entrez.read(handle)
    pmids = records['IdList']
    webenv = ""
    query_key = ""
    pmids_url = ""
    for pmid in pmids:
        pmids_url += pmid + ","
    abstract_url = f'http://eutils.ncbi.nlm.nih.gov/entrez//eutils/efetch.fcgi?db=pubmed&id={pmids_url}'
    abstract_ = urllib.request.urlopen(abstract_url).read().decode('utf-8')
    abstract_bs = BeautifulSoup(abstract_,features="xml")
    articles_iterable = abstract_bs.find_all('PubmedArticle')
    abstract_texts = [x.find('AbstractText').text for x in articles_iterable]
    
    output_dict = {}
    for pmid, abt in zip(pmids, abstract_texts):
        output_dict[pmid] = abt
    final_output_dict = {"abstracts":output_dict, "terms": terms}
    json.dump(final_output_dict, out_handle)
    out_handle.close()

def fetch_pmc_ids(term, literature_dir):
    """
    Given a list of terms, fetch PMC ids for free articles associated with the search terms and write it to a text file.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    Entrez.email = "caceves@scripps.edu"

    term_filename = os.path.join(literature_dir, "{term}.txt".format(term=term.replace(" ","_")))
    if os.path.isfile(term_filename):
        return(0)
    out_handle = open(term_filename, "w")
    handle = Entrez.esearch(db="pubmed", term=term + " AND fha[Filter]", retmode="xml", retmax=500, sort="relevance", usehistory='n')
    records = Entrez.read(handle)
    print(records)
    pmids = records['IdList']
    webenv = ""
    query_key = ""
    print(pmids)
    abstract_url = f'http://eutils.ncbi.nlm.nih.gov/entrez//eutils/efetch.fcgi?db=pubmed&id={pmids}'
    abstract_ = urllib.request.urlopen(abstract_url).read().decode('utf-8')
    abstract_bs = BeautifulSoup(abstract_,features="xml")
    articles_iterable = abstract_bs.find_all('PubmedArticle')
    # Abstracts
    abstract_texts = [ x.find('AbstractText').text for x in articles_iterable]
    print(abstract_texts[0])
    print(len(pmids))
    print(len(abstract_texts))
    sys.exit(0)
    #write all pmids to the list
    for p in pmids:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", linkname="pubmed_pmc", id=p, retmode="text")
        return_handle = Entrez.read(handle)
        handle.close()
        try:
            pmc  = "PMC" + str(return_handle[0]['LinkSetDb'][0]['Link'][0]['Id'])
        except:
            return(0)
        out_handle.write(pmc)
        out_handle.write("\n")
    out_handle.close()
    return(0)
