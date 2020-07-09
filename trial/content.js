// script that runs when page loads

// NOTE: should have function that checks if only ONE email is open 
// and if not then popup saying "plz open email to scan"
// NOTE 2: popup is html file!!

// TODO: on click of the extension button << grayscale logo when not active color when active??
// calls in from info-popup.html file

// TODO: get email address

// TODO: if email open >> fetch message id from html elements
// let messageId = document.getElementById("");

let messageId;

function get_msg (){
  message = document.getElementById('raw_message_text');
  alert(message);
  nn_js();
}
// TODO: NN determines ham or spam  
function nn_js(){
    //call neural net with message
    alert("in progress");
    import * as tf from '@tensorflow/tfjs';

    // const model = await tf.loadLayersModel('file_busters_js_7.h5');

    // const prediction = model.preprocess_input_email(msg);
    let prediction = 'spam';

    // TODO: if spam >> alert user of spam (bright red warning sign)
    if (prediction === 'spam'){
        alert("ALERT: SPAM DETECTED! Be careful of any links, requests, or attachments in this email!")
    }
    // TODO: if ham >> alert user of ham (smiley face??)
    else if (prediction === 'not spam'){
        alert("All good here! No spam detected")
    }
    else {
        alert("Hmmm...we couldn't determine if this email is spam.")
    }
}

setTimeout(get_msg, 1000);
// TODO: if spam >> alert user of spam (bright red warning sign)

// TODO: if ham >> alert user of ham (smiley face??)