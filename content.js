// script that runs when page loads

// NOTE: should have function that checks if only ONE email is open 
// and if not then popup saying "plz open email to scan"
// NOTE 2: popup is html file!!

// TODO: on click of the extension button << grayscale logo when not active color when active??

// TODO: get current tab 

// TODO: check if single email is open

// TODO: if email open >> fetch message id from html elements

// TODO: pass message id to gmail api to get email data 

/**
 * Get Message with given ID.
 *
 * @param  {String} userId User's email address. The special value 'me'
 * can be used to indicate the authenticated user.
 * @param  {String} messageId ID of Message to get.
 * @param  {Function} callback Function to call when the request is complete.
function getMessage(userId, messageId, callback) {
    var request = gapi.client.gmail.users.messages.get({
      'userId': userId,
      'id': messageId
    });
    request.execute(callback);
  }
*/

// TODO: NN determines ham or spam  

// TODO: if spam >> alert user of spam (bright red warning sign)

// TODO: if ham >> alert user of ham (smiley face??)