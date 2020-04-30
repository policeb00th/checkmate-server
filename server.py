from flask import Flask, redirect, url_for, request,jsonify
import cm
app = Flask(__name__)

@app.route('/success/Checkmate')
def success(name):
   return cm.abc(name)

@app.route('/Checkmate',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['pdf']
      a=cm.abc(user)                                     #user will be a string containing all the stuff in the pdf that was uploaded
      b={1:a}
      return jsonify(b)
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))
if __name__ == '__main__':
   app.run(debug = True,port=8080)