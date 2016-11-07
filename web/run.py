from flask import Flask, render_template, request
import datetime

app = Flask(__name__)

@app.template_filter()
def datetimefilter(value, format='%Y/%m/%d %H:%M'):
    """convert a datetime to a different format."""
    return value.strftime(format)

app.jinja_env.filters['datetimefilter'] = datetimefilter

@app.route("/")
def template_test():
    return render_template('layout.html', my_string="Wheeeee!", 
        title="Index") 

@app.route("/image", methods=['GET'])
def login():
    location = request.args.get('pic')
    return render_template('layout.html', my_string=location, 
        title="Index") 

if __name__ == '__main__':
    app.run(debug=True)
