from typing import Type
from user_input_process import user_input_init
import pandas as pd
import os
current_dir = os.getcwd()

#found the similar question in the complex QA list
class Prompt_Fewshot:
    def __init__(self, user_input:Type[user_input_init], foundqadf):
        self.user_input_cls = user_input
        self.foundqadf = foundqadf
    
    def compile_examples(self):
        examples = ""
        for i in range(self.foundqadf.shape[0]):
            q = 'Question: ' +self.foundqadf['question'][i]
            a = 'SQL: '+self.foundqadf['sql'][i]
            examples = examples+ '\n'.join([q,a])
        return(examples)


        
    
    def compile(self):
        examples = self.compile_examples()
        hints = self.user_input_cls.gen_hints()
        user_question = self.user_input_cls.input

        prompt = f"""
#Character#
You are an expert of SQL language and the best skill of you is mimic similar SQL statements to write new SQL statements.
And you can replace the special terminologies and time points in SQL according to the user question.

#Task#
Now, write a SQL statement to answer the user question, modeled after the following Examples.

#Limitations#
1. Your SQL must use the terminologies given by "HINTS", DO NOT change the terminologies themselives.
2. Your SQL must imitate the Examples to be grammerly correct.
3. Your SQL need to be readable, so enter line breaks where appropriate. 
4. Your SQL should be careful on the dependency of fields you use.
5. Make sure that the your SQL can be executed by pd.read_sql_query().


#Examples#
{examples}


#Now Write SQL#
Question: {user_question}
HINTS: {hints}
SQL:
        """

        return prompt



# DIDNOT found the similar question in the complex QA list
class Prompt_Zeroshot:
    def __init__(self, user_input:Type[user_input_init]):
        self.user_input_cls = user_input
        self.meta = pd.read_csv(os.path.join(current_dir,'backend/dbmata.csv'), index_col=0)

    def indentify_table_fields(self):

        fields = set()
        
        #part1: fields found by hints: [ [term, field, field_des] ]
        hintslist = self.user_input_cls.hintslist
        if len(hintslist)>0:
            fields = fields|set([ h[1] for h in hintslist ])
        
        #part2: fields found in meta,  
        for word in self.user_input_cls.input.split():
            for i in range(self.meta.shape[0]):

                line = list(self.meta.iloc[i])

                linestr = ','.join([str(j).lower() for j in line])
                if word.lower() in linestr:
                    fields = fields|set([ line[2] ])
        
        return(list(fields))
        
    
    def compile_meta(self):

        fields = self.indentify_table_fields()
        print(fields)
        # fields -map> meta
        submeta = self.meta[self.meta['Column'].isin(fields)]
        print(submeta)
        # meta -map> table
        tables = set(submeta['Table'])
        print(tables)
        # full table: fields    
        submeta = self.meta[self.meta['Table'].isin(tables)]
        print(submeta)
        # complie meta str for prompt
        schema = []
        for table in tables:

            sub = self.meta[self.meta['Table'] == table].fillna('')
            tabledes = ' '.join(set(sub['Table Description']))
            schema.append('Table: ' + table + '\tComment: ' + tabledes)
            schema.append("Has Fields:")
            for i in range(sub.shape[0]):
                 field = sub.iloc[i]['Column']
                 schema.append(field + '\tComment: ' + sub.iloc[i]['Column Description'])
            schema.append('\n')


        return '\n'.join(schema)

    def compile(self):
        
        user_question = self.user_input_cls.input
        hints = self.user_input_cls.gen_hints()
        if len(hints) < 5:
            hints = 'No Hints for this question'
        schema = self.compile_meta()
        if len(schema) < 5:
            schema = 'Did not find any ralted table from the database. Just Write the SQL according to your understandings.'

        prompt = f"""
#Character#
You are an expert of SQL language and you serve in a world-class company. You are familiar with the tables and fields in the company's database. 
Therefore your job is answer the data retrival queries from the staff using SQL. 

#Task#
Now, you have a question to solve, the staff also told you the terminologies in this question are related to which tables and fields. 
You need to utilze the following table schema information to write a correct SQL.

#Limitations#
1. Your SQL must use the terminologies given by "HINTS", DO NOT change the terminologies themselives.
2. Your SQL need to be readable, so enter line breaks where appropriate.
3. Your SQL must be grammerly correct, so be careful on the dependency of fields you use.
4. You can only write one SQL statement, NO ONE need extra explainations.
5. Make sure that the your SQL can be executed by pd.read_sql_query().


#Schema#
Tables: 
{schema}


#Now Write SQL#
Question: {user_question}
HINTS: {hints}
SQL:
"""

        return prompt



class Prompt_Answer:
    def __init__(self, taskinfos) -> None:
        self.taskinfos = taskinfos


    def complie(self):

        prompt = f"""
#Character#
Now, your are a Text2SQL task assistant. Your tone is objective and instructive. 
You are good at summaring the given #Task WorkingFlow# to the user. 

#Background#
You completed a Text2SQL task through the following steps: 
    - process the user question; 
    - search in the pre-stored question-SQL pairs and find similar pairs; 
    - if similar pairs are found, imitate and write the SQL, or wirte the SQL by yourself;  
    - execute the SQL you write, and get the data.

#Limitations#
1. Give appropriate summary about the structure of the SQL statement
3. Must Give the answer to the user question.
2. Keep the word count to 40 words or less

Now, summary the following WorkingFlow to your user:
#Task WorkingFlow#
User Question: {self.taskinfos['question']}
Question-SQL pairs: {self.taskinfos['search_res']}
SQL: 
{self.taskinfos['sql']}
SQL execution (only the first 3 rows): 
{self.taskinfos['sqlexe']}
"""

        
        return prompt
    

if __name__ == "__main__":

    from similarity_cal import search_qvec

    emb = {"name":"m3e-base-local","url":"http://10.29.253.101:19997/v1/embeddings","token":""}


    # few-shot
    # user_input = 'Did Blake book Adan Dinning?'
    # full_processed_input = user_input_init(user_input).full_process()
    # _, _, foundqadf = search_qvec(full_processed_input,emb)
    # p = Prompt_Fewshot(full_processed_input, foundqadf).compile()

    # self-gen
    user_input = 'How many Booking records are there?'
    #user_input = 'aaaa'
    full_processed_input = user_input_init(user_input).full_process()
    p = Prompt_Zeroshot(full_processed_input).compile()

    # answer
    # user_input = 'Did Blake book Adan Dinning?'
    # full_processed_input = user_input_init(user_input).full_process()
    # taskinfos = {
    #     'question':full_processed_input.input,
    #     'search_res':'aaaaaaaaaaaaaaaasearch_res',
    #     'sql':'bbbbbbbbbbbbbbbbbbbbbbb',
    #     'sqlexe':pd.DataFrame(columns= ['Booking_ID', 'Customer_ID', 'Workshop_Group_ID'])
    # }
    # p = Prompt_Answer(taskinfos).complie()

    print(p)