import re
from datetime import datetime
import pandas as pd
import os
current_dir = os.getcwd()

class user_input_init():
    def __init__(self,input):
        self.input = input
        self.trans = input
        self.hintslist = []

        self.time_replace = False
        self.term_replace = False

    def time_recognition(self):

        today = [int(i) for i in datetime.now().date().strftime('%Y-%m-%d').split('-')] #[2024, 11, 21]

        dd_patterns = [
            r'(\d{2,4}-\d{1,2}-\d{1,2})',  # YYYY-MM-DD
            r'(\d{2,4}/(\d{1,2})/\d{1,2})',  # YYYY/MM/DD
            r'(\d{2,4}\\\d{1,2}\\\d{1,2})',  # YYYY\MM\DD
            r'(\d{2,4}\.\d{1,2}\.\d{1,2})',  # YYYY.MM.DD
            r'(\d{2,4}年\d{1,2}月\d{1,2}日)' # YYYY年MM月DD日
            #r'(\d{1,2})月(\d{1,2})日', # YYYY年MM月DD日
            #r'(\d{1,2})日', # YYYY年MM月DD日
            # r'今日', # DD日
            # r'昨日', # DD日
            # r'明日', # DD日
            # r'今天', # DD日
            # r'昨天', # DD日
            # r'明天' # DD日
        ]
        combined_dd_pattern = '|'.join(pattern for pattern in dd_patterns)

        mm_patterns = [
            r'(\d{2,4}-\d{1,2})',  # YYYY-MM
            r'(\d{2,4}/\d{1,2})',  # YYYY/MM
            r'(\d{2,4}\\\d{1,2})',  # YYYY\MM
            r'(\d{2,4}\.\d{1,2})',  # YYYY.MM
            r'(\d{2,4}年\d{1,2}月)'  # YYYY年MM月
            #r'((\d{1,2})月)', # MM月
            # r'本月', # MM月
            # r'这个月', # 
            # r'上个月', # 
            # r'下个月'
        ]
        combined_mm_pattern = '|'.join(pattern for pattern in mm_patterns)

        #先找日期
        dd_dates = []
        matches = re.findall(combined_dd_pattern, self.input)
        for match in matches:
            match = list([x.strip() for x in match if x.strip()!=''])[0]
            dd_dates.append(match)
        
        for dd in dd_dates:
            self.trans = self.trans.replace(dd, '某日期')

        #再找月份
        mm_dates = []
        matches = re.findall(combined_mm_pattern, self.trans)
        for match in matches:
            match = list([x.strip() for x in match if x.strip() != ''])[0]
            mm_dates.append(match)

        for mm in mm_dates:
            self.trans = self.trans.replace(mm, '某月份')
        print('替换日期：', self.trans)

        self.time_replace = True
        return self

    def term_recognition(self):
 
        term_df = pd.read_csv(os.path.join(current_dir,"backend/term.csv"))
        field_df = pd.read_csv(os.path.join(current_dir,'backend/dbmata.csv'))

        for term in term_df['term']:
            if term in self.trans:
                field = term_df[term_df['term'] == term]['field'].values[0]
                try:
                    field_des = field_df[field_df['Column'] == field]['Column Description'].values[0]
                except:
                    field_des = field
                self.hintslist.append([term,field,field_des])
                print('...FOUND TERM ', term, field_des)
                self.trans = self.trans.replace(term, "["+field_des+']')

        print('...term_recognition SUCCESS, MASkED：',self.trans)

        self.term_replace = True
        return self
    
    def gen_hints(self):  ## gen hint sentence
        if self.term_replace == False:
            self.term_recognition()
        hint_str = ''
        for item in self.hintslist:
            hint_str += f"[{item[0]}] bolongs to field [{item[1]}], means {item[2]}; "
        return(hint_str)
    
    def full_process(self):
        print("...PROCESSING THE USER INPUT...")
        return self.term_recognition()#.time_recognition()

if __name__ == '__main__':
    input = user_input_init("Did Blake book Adan Dinning?")
    input.full_process()
    hints = input.gen_hints()

    print(input.hints, hints)