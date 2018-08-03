#!/usr/bin/env python3

from flask import Flask, redirect, render_template, request, url_for, flash
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import string, json, datetime

#vader sentiment analysis
analyzer = SentimentIntensityAnalyzer()
translator = str.maketrans('', '', string.punctuation)

#load thirsty dictionary
df = pd.read_csv("dictionary.csv", header=None)
dictionary = df.values.T.tolist()[0]

#variables
comments, thirstScore = [], []
sessionMood, thirst_to_send= 0.0, 0

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main():

    dirtyCount = 0
    sessionMood = np.mean(thirstScore)

    # mood ring hsl changing
    s, l = 100, 50
    if np.isnan(sessionMood):
        h = 300
    else:
        h = (((sessionMood+0.7)*100)/1.4)+250

    color = "hsl({}, {}%, {}%)".format(str(h), str(s), str(l))

    #if GET: render website with comments
    if request.method == "GET":
        return render_template("index.html", comments=comments, moods=thirstScore, sessionMood=sessionMood, color=color)

    #if POST: add comment to list and calculate mood score
    comment_raw = request.form["contents"]
    comment = comment_raw.translate(translator)
    comments.append(comment_raw)

    # get sentiment score
    vs = analyzer.polarity_scores(comment)
    mood = vs["compound"]

    # get dirty score
    commentVec = comment.split()
    dirtyWords = list(set(dictionary) & set(commentVec))

    if len(dirtyWords) == 1:
        dirtyCount = 1
    elif len(dirtyWords) >= 10:
        dirtyCount = 4
    elif len(dirtyWords) >= 5:
        dirtyCount = 3
    elif len(dirtyWords) >= 2:
        dirtyCount = 2

    # thirst = sentiment + dirty
    thirst = abs(mood + dirtyCount)
    print("Thirst score: ", mood, "+", len(dirtyWords), dirtyCount)


    # send thirst score
    global thirst_to_send
    global currentTime
    thirst_to_send = int(np.ceil(thirst))
    currentTime = str(datetime.datetime.now().time())
    thirstScore.append(thirst)

    return redirect(url_for("main"))

@app.route('/score')
def getdata():
    global thirst_to_send
    global currentTime
    content = {'event': 'ThirstData', 'data': str(thirst_to_send), 'published_at': currentTime}
    return(json.dumps(content))

if __name__ == "__main__":
    app.config["DEBUG"] = True
    # app.run(host="0.0.0.0", port=80)
    app.run()
