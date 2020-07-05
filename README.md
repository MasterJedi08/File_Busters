# File_Busters
AcademyNEXT 2020 - File_Busters Chrome extension

<h3>What is it? </h3>
Source code for the Chrome extension File_Busters
<br>File_Busters scans open emails and detects any content that could potentially be spam

<h3>Project Contributors: </h3>
Github users: 
<ul style="list-style-type:none;">
  <li><a href ="https://github.com/MasterJedi08">MasterJedi08</a></li>
  <li><a href = "https://github.com/captain-snuffles">captain-snuffles</a></li>
  <li><a href = "https://github.com/jazzy0131">jazzy0131</a></li>
</ul>

<h3>Available Browsers</h3>
Chrome

<h3>Files and What They Do</h3>
  <h5>Extension Files</h5>
  manifest.json - required manifest for extensions
  content.js - JavaScript file of script that runs when active page loads
  background.js - JavaScript file of script that runs in background when browser opens
    <br></br>
  <h5>Neural Network Python Files</h5>
  n6.py - the neural network code
  nn.py - the original neural network (not used)
  file_busters_model.h5 - the neural network model as an h5 file
  files.py - wrote the original emails to text files after data cleaning
  clean.py - cleans data
  email_files folder - contains the cleaned text files of ham and spam emails

<h3>Known Errors: </h3>
