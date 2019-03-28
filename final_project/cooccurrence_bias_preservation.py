import json
import logging
from log import init_console_logger
import argparse
import os
from  nltk import ngrams
import multiprocessing as mp


LOGGER = logging.getLogger('bias scores')
LOGGER.setLevel(logging.DEBUG)

#yusu added words from the rule file
DEFAULT_MALE_NOUNS = {
    'Countryman',
    'fraternal','wizards','manservant','fathers','divo','actor','bachelor','papa',
    'dukes','barman','countrymen','brideprice','hosts','potential_suitors','airmen','andropause',
    'penis','prince','governors','abbot','men','widower','gentlemen','sorcerers','sir',
    'bridegrooms','baron','househusbands','gods','nephew','widowers','lord','brother',
    'grooms','priest','adultors','andrology','bellboys','his','marquis','princes','emperors',
    'stallion','chairman','monastery','priests','boyhood','fellas','king','dudes',
    'daddies','manservant','semen','spokesman','tailor','cowboys','dude','bachelors','barbershop','emperor','daddy',
    'masculism','guys','enchanter','guy','fatherhood','androgen','cameramen','godfather',
    'strongman','god','patriarch','uncle','chairmen','sir','brotherhood','host','testosterone',
    'husband','dad','steward','males','cialis','spokesmen','pa','beau','stud','bachelor',
    'wizard','sir','nephews','fathered','bull','beaus','councilmen','landladies','grandson',
    'fiances','stepfathers','horsewomen','grandfathers','adultor','schoolboy','rooster','grandsons',
    'bachelor','cameraman','dads','him','master','lad','policeman','monk','actors','salesmen',
    'boyfriend','councilman','fella','statesman','paternal','chap','landlord','brethren','lords',
    'blokes','fraternity','bellboy','duke','ballet_dancer','dudes','fiance','colts',
    'husbands','suitor','maternity','he','businessman','masseurs',
    'hero','deer','busboys','boyfriends','kings','brothers','masters','stepfather','brides',
    'son','studs','cowboy','mentleman','sons','baritone','salesman','paramour','male_host',
    'monks','menservants',"mr.",'headmasters','lads','congressman','airman','househusband',
    'priest','barmen','barons','abbots','handyman','beard','fraternities','stewards','colt',
    'czar','stepsons','himself','boys','lions','gentleman','his','masseur','bulls','uncles','bloke','beards',
    'hubby','lion','sorcerer','macho','father','gays','male','waiters','stepson','prostatic_utricle',
    'businessmen','heir','waiter','headmaster','man','governor','god','bridegroom','grandpa',
    'groom','dude','gay','gents','boy','grandfather','gelding','paternity',
    'roosters','prostatic_utricle','priests','manservants','stailor','busboy','heros'
}

#yusu added special
SPECIAL_MALE_NOUNS = {
    'Sperm','Beard','Mustache',
    'Ejaculation','Erection','Scrotum','Penis','Testicles','Epididymis'
}

#yusu added words from the rule file
DEFAULT_FEMALE_NOUNS = {
    'woman', 'women', 'ladies', 'female', 'females', 'girl', 'girlfriend',
    'girlfriends', 'girls', 'her', 'hers', 'lady', 'she', 'wife', 'wives','countrywoman',
'sororal','witches','maidservant','mothers','diva','actress','spinster','mama',
    'duchesses','barwoman','countrywomen','dowry','hostesses','airwomen','princess',
    'governesses','abbess','women','widow','ladies','sorceresses','madam','brides',
    'baroness','housewives','godesses','niece','widows','lady',
'sister','brides','nun','adultresses','obstetrics','bellgirls','her','marchioness',
    'princesses','empresses','mare','chairwoman','convent','priestesses','girlhood',
    'ladies','queen','gals','mommies','maid','female_ejaculation','spokeswoman','seamstress',
    'cowgirls','chick','spinsters','empress','mommy','feminism','gals','enchantress','gal','motherhood','estrogen'
    'camerawomen','godmother','strongwoman','goddess','matriarch','aunt','chairwomen',"ma'am",
    'sisterhood','hostess','estradiol','wife','mom','stewardess','females','viagra',
    'spokeswomen','ma','belle','minx','maiden','witch','miss','nieces','mothered',
    'cow','belles','councilwomen','landlords','granddaughter','fiancees','stepmothers',
    'horsemen','grandmothers','adultress','schoolgirl','hen','granddaughters','bachelorette',
    'camerawoman','moms','her','mistress','lass','policewoman','nun','actresses',
    'saleswomen','girlfriend','councilwoman','lady','stateswoman','maternal',
    'lass','landlady','sistren','ladies','wenches','sorority','bellgirl','duchess',
    'ballerina','chicks','fiancee','fillies','wives','suitress','paternity','she',
    'businesswoman','masseuses','heroine','doe','busgirls','girlfriends','queens',
    'sisters','mistresses','stepmother','grooms','daughter','minxes','cowgirl',
    'lady','daughters','mezzo','saleswoman','mistress','hostess','nuns','maids',"mrs.",
    'headmistresses','lasses','congresswoman','airwoman','housewife','priestess','barwomen',
    'barnoesses','abbesses','handywoman','toque','sororities','stewardesses',
    'filly','czarina','stepdaughters','herself','girls','lionesses','lady','vagina','hers','masseuse','cows','aunts',
    'wench','toques','wife','lioness','sorceress','effeminate','mother','lesbians',
    'female','waitresses','skene_gland','stepdaughter','womb','businesswomen',
    'heiress','waitress','headmistress','woman','governess','godess','bride','grandma','bride','gal','lesbian','ladies','girl',
    'grandmother','mare','maternity','hens','nuns','maidservants','seamstress','busgirl','heroines'
}


SPECIAL_FEMALE_NOUNS = {
    'dowry',
    'womb','pregnant','pregnancy','breast','abortion','miscarriage','clitoris',
    'menopause','vagina','uterus','Ovaries'
}
DEFAULT_TARGET_POS = {'VERB', 'ADJ', 'ADV'}


def sortbybias(d):
    
    d_s = sorted(d.items(), key = lambda t: t[1] )
    return d_s

def gender_ratios_m_f(output_data_dir,file):
    n = 0
    tot = 0 
    print("Gender Ratios...")
    with open(file,'r') as f:
        data = json.load(f)
    bias_record = {}
    #yusu edited the next line
    for words in data:
        if words not in DEFAULT_MALE_NOUN and words not in DEFAULT_FEMALE_NOUN and words not in DEFAULT_MALE_SPECIAL and words not in DEFAULT_FEMALE_SPECIAL:

            if (data[words]['m']+data[words]['f']!=0 and data[words]['f']!=0 and data[words]['m']!=0):
                print(data[words])
                #yusu edited the next line
                score1 = abs((data[words]['m']-data[words]['f'])/(data[words]['m']+data[words]['f']))

                tot+=score1
                n +=1
                rec = {"b_score" : score1}
                data[words].update(rec)
                bias_record[words] = json.dumps(data[words])
    # print(bias_record)
    # print(sortbybias(bias_record))
    output_file = os.path.join(output_data_dir, 'biased_words_m_f')   
    print("Bias_score: ", (tot/n))
    with open(output_file,'w') as fp:
        json.dump(bias_record,fp, sort_keys=True)   



def gender_ratios(output_data_dir,file):
    print("Gender Ratios...")
    with open(file,'r', encoding='utf-8',
                 errors='ignore') as f:
        data = json.load(f)
    bias_record = {}
    #yusueditsthe next line
    for words in data:
        if words not in DEFAULT_MALE_NOUN and words not in DEFAULT_FEMALE_NOUN and words not in DEFAULT_MALE_SPECIAL and words not in DEFAULT_FEMALE_SPECIAL:
            #yusu's edit starts
            if (data[words]['m']+data[words]['f']!=0):
                score1 = abs((data[words]['m']-data[words]['f'])/(data[words]['m']+data[words]['f']))
                rec = {"b_score" : score1}
                data[words].update(rec)
                bias_record[words] = json.dumps(data[words])
    # print(bias_record)
    # print(sortbybias(bias_record))
    output_file = os.path.join(output_data_dir, 'biased_words')    
    with open(output_file,'w') as fp:
        json.dump(bias_record,fp, sort_keys=True)   
 #yusu adds this chunk       
def preservation_ratios(output_data_dir,file):
    print("preservation Ratios...")
    with open(file,'r', encoding='utf-8',
                 errors='ignore') as f:
        data = json.load(f)
    preserve_record = {}
    #yusueditsthe next line
    for words in data:
        if words in DEFAULT_MALE_NOUN or words in DEFAULT_FEMALE_NOUN or words in DEFAULT_MALE_SPECIAL or words in DEFAULT_FEMALE_SPECIAL:
            #yusu's edit starts
            if (data[words]['m']+data[words]['f']!=0):
                score2 = abs((data[words]['m']-data[words]['f'])/(data[words]['m']+data[words]['f']))
                rec = {"p_score" : score2}
                data[words].update(rec)
                preserve_record[words] = json.dumps(data[words])
    # print(bias_record)
    # print(sortbybias(bias_record))
    output_file = os.path.join(output_data_dir, 'preserved_words')    
    with open(output_file,'w') as fp:
        json.dump(preserve_record,fp, sort_keys=True)        

def preservation_ratios_m_f(output_data_dir,file):
    n = 0
    tot = 0 
    print("Preservation Ratios...")
    with open(file,'r') as f:
        data = json.load(f)
    preserve_record = {}
    #yusu edited the next line
    for words in data:
        if words in DEFAULT_MALE_NOUN or words in DEFAULT_FEMALE_NOUN or words in DEFAULT_MALE_SPECIAL or words in DEFAULT_FEMALE_SPECIAL:
            if (data[words]['m']+data[words]['f']!=0 and data[words]['f']!=0 and data[words]['m']!=0):
                print(data[words])
                #yusu edited the next line
                score2 = abs((data[words]['m']-data[words]['f'])/(data[words]['m']+data[words]['f']))

                tot+=score2
                n +=1
                rec = {"p_score" : score2}
                data[words].update(rec)
                preserve_record[words] = json.dumps(data[words])
    # print(bias_record)
    # print(sortbybias(bias_record))
    output_file = os.path.join(output_data_dir, 'preserved_words_m_f')   
    print("Preservation_score: ", (tot/n))
    with open(output_file,'w') as fp:
        json.dump(preserve_record,fp, sort_keys=True)  

def get_cooccurrences(file, data, window):           
    
   
    with open(file, 'r', encoding='utf-8',
                 errors='ignore') as fp:
        # print(fp)
        sentences = fp.read()

    male_nouns = DEFAULT_MALE_NOUNS
    female_nouns = DEFAULT_FEMALE_NOUNS
    #yusu's edit starts:
    male_special = SPECIAL_MALE_NOUNS
    female_special = SPECIAL_FEMALE_NOUNS
    #yusu's edit ends
    n_grams = ngrams(sentences.split(), window)
    
    for grams in n_grams:
        pos = 1
        m = 0 
        f = 0 
        #yusu's edit starts:
        m_s = 0
        f_s = 0
        #yusu's edit ends
        for w in grams:
                pos+=1
                if w not in data:
                    data[w]= {"m":0, "f":0}
                
                if pos==int((window+1)/2):
                    if w in male_nouns:
                        m = 1
                    if w in female_nouns:
                        f = 1
                        #yusu's edit starts
                    if w in male_special:
                        m_s = 1
                    if w in female_special:
                        f_s = 1
                        #yusu's edit ends
                    if m > 0:
                        for t in grams:
                            if t not in data:
                                data[t]= {"m":0, "f":0, "m_s":0, "f_s":0}
                            data[t]['m']+=1
                    if f > 0:
                        for t in grams:
                            if t not in data:
                                data[t]= {"m":0, "f":0,"m_s":0, "f_s":0 }
                            data[t]['f']+=1
                            #yusu's edit starts
                    if m_s > 0:
                        for t in grams:
                            if t not in data:
                                data[t]= {"m":0, "f":0, "m_s":0, "f_s":0}
                            data[t]['m_s']+=1
                    if f_s > 0:
                        for t in grams:
                            if t not in data:
                                data[t]= {"m":0, "f":0, "m_s":0, "f_s":0}
                            data[t]['f_s']+=1
                            #yusu's edit ends
    return data
    

def coccurrence_counts(dataset_dir, output_dir, window=7,num_workers=1):
    
    
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)
    output_data_dir = os.path.join(output_dir, 'bias_scores')
    
    if not os.path.isdir(dataset_dir):
        raise ValueError('Dataset directory {} does not exist'.format(dataset_dir))

    if not os.path.isdir(output_data_dir):
        os.makedirs(output_data_dir)
        
    data ={}
    worker_args = []
    LOGGER.info("Getting list of files...")
    for root, dirs, files in os.walk(dataset_dir):
        root = os.path.abspath(root)
        for fname in files:
            basename, ext = os.path.splitext(fname)
            if basename.lower() == 'readme':
                continue
            txt_path = os.path.join(root, fname)
            data = get_cooccurrences(txt_path, data, window )
    output_file = os.path.join(output_data_dir, 'all_words')     

    with open(output_file,'w') as fp:
        json.dump(data,fp)
    
    gender_ratios(output_data_dir,output_file)  
    gender_ratios_m_f(output_data_dir,output_file) 
    #yusu adds
    preservation_ratios(output_data_dir,output_file)  
    preservation_ratios_m_f(output_data_dir,output_file) 
    
    #yusu's edit starts

            
            
def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Get the bias scores of a given text file')
    parser.add_argument('dataset_dir', help='Path to directory containing text files', type=str)
    parser.add_argument('output_dir', help='Path to output directory', type=str)
    parser.add_argument('-n', '--num-workers', dest='num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('-w', '--window', dest='window', type=int, default=10, help='Context Window')
    return vars(parser.parse_args())


if __name__ == '__main__':
    init_console_logger(LOGGER)
    coccurrence_counts(**(parse_arguments()))