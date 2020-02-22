#!/usr/bin/python3.5
from bottle import *
import controller

app = default_app()
run(host='127.0.0.1', port=2400, reload=True)
