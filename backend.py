from flask import Flask, request
from flask_cors import CORS
from groq import Groq
from io import StringIO

import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')


from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import re



app = Flask(__name__)
CORS()



def NLP(input):

    input = input.lower()
    input = re.sub(r'[^\w\s]', '',input)
    input = re.sub(r'\d','',input)

    tokens = tokenize.word_tokenize(input, language='portuguese')

    stopWords = set(stopwords.words('portuguese'))

    tokens_filtered = [token for token in tokens if token not in stopWords]
    stemmer = RSLPStemmer()

    tokens_stem = [stemmer.stem(token) for token in tokens_filtered]

    return " ".join(tokens_stem)


def handlePDF(pdf):
    output_string = StringIO()
    parser = PDFParser(pdf)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
    return output_string.getvalue()


@app.route("/Content", methods=['POST'])
def HandleContent():

    if 'file' in request.files:
        file = request.files['file']
        if file.content_type == "application/pdf":
            content = handlePDF(file)
        else:
            content = file.read().decode('utf-8')
    else:
        content = request.data.decode('utf-8')
        print(content)
    content = NLP(content)
    #print(content)

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {

                "role" : "system",
                "content": (
                "Você vai receber um e-mail e deve classificar ele como produtivo ou improdutivo. "
                "Um e-mail produtivo: Emails que requerem uma ação ou resposta específica "
                "(ex.: solicitações de suporte técnico, atualização sobre casos em aberto, dúvidas sobre o sistema). "
                "Um e-mail improdutivo: Emails que não necessitam de uma ação imediata "
                "(ex.: mensagens de felicitações, agradecimentos). Depois gere uma resposta para esse e-mail."
            )

            },
            {
                "role": "user",
                "content": content
            }
        ],
        model="llama3-8b-8192"
    )


    return {'message': chat_completion.choices[0].message.content }, 200


