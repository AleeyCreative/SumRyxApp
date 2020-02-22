def whitelist_check(words):
    words = [word for word in words if word not in ['Allah','God']]
    words = [word for word in words if word not in ['MAN','WHO','RAM']]
    return words
