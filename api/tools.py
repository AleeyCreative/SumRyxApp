def parseDocument(postObj,n):
    docts = []
    for i in range(1, n):
        fileObj = postObj['file' + str(i)]
        filename = fileObj.raw_filename
        file_extension = filename.split(".")[1]
        if file_extension not in ['pdf','txt']:
            return False
        contents = ""
        if (file_extension == 'pdf'):
            contents = parsePDF(fileObj.file)
            docts.append({'title':filename, 'text':contents})
        elif (file_extension == 'txt'):
            for line in fileObj.file.readlines():
                contents += line.decode()
            docts.append({'title':filename, 'text':contents})
    return docts



def parsePDF(file):
	return False
