// script that runs when page loads

// NOTE: should have function that checks if only ONE email is open 
// and if not then popup saying "plz open email to scan"
// NOTE 2: popup is html file!!

chrome.runtime.onMessage.addListener(gotMessage);

function gotMessage(message, sender, sendResponse) {
    console.log(message.txt);

}