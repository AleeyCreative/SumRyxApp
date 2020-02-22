from bottle import *
from summarizer import summarizeDocument
import json
import numpy as np
from tools import *


@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = "*"
    response.headers['Access-Control-Allow-Methods'] = "PUT, GET, POST, DELETE"
    response.headers['Access-Control-Allow-Headers'] = "Authorization, X-Access-Token, Content-Type"
    pass

@get('/')
def index():
    response.headers['Content-Type'] = 'application/json'
    print("hi there people")
    return {'msg':'Welcome to Sumryx, A summarization application'}

@route('/summarize', method="POST")
def summ_handler():
    print("New file recieved !!")
    n = int(request.POST['count'])
    contents = parseDocument(request.POST, n)
    if(contents == False):
			return {"error":"Document Format unsupported yet !!"}
		else:
			print(contents)
			summaries = [summarizeDocument(content['text'], content['title']) for content in contents]
			print(summaries)
			response.headers['Content-Type'] = 'application/json'
			return {'done':True, 'summaries':summaries}


@get('/file')
def sendMessage():
    return static_file('file1.txt', root='./summaries', download=True)

@post('/download-summaries')
def downloadSummaries():
    summaries = request.POST['summaries']
    print(summaries)
    file_name = write_summaries(summaries)
    return static_file(file_name, root='./summaries', download=True)
