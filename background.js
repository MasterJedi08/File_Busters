// script runs as soon as chrome launches
// listens for certain "events"
// ^^ can listen for "browser action" (ex: when button is pressed)

//script that makes it occur when extension clicked
chrome.browserAction.onClicked.addListener(buttonClicked);

//what happens when listener detects button clicked
function buttonClicked(tab) {
    //message as object
    let msg = {
        txt: "hello"
    }

    // sends message to content.js
    chrome.tabs.sendMessage(tab.id, msg);
}