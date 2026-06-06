# LinguaSign Verification & Testing Checklist

A structured, step-by-step checklist to help you verify all core features on both the web and mobile app.

---

## 🔐 1. Authentication & Security

### User Story: User Registration
*   [ ] **Action**: Create a new account with a unique email and password.
*   [ ] **Verification**: App automatically logs in upon registration and displays the Home page with the user's dropdown menu.

### User Story: Secure User Login
*   [ ] **Action**: Attempt to log in with an incorrect email.
*   [ ] **Verification**: The app displays `"Invalid email or password"`.
*   [ ] **Action**: Attempt to log in with a correct email but an incorrect password.
*   [ ] **Verification**: The app displays `"Invalid email or password"` (confirming that specific email/password validation details are hidden for security).
*   [ ] **Action**: Log in with correct credentials.
*   [ ] **Verification**: User is logged in successfully.

---

## 👤 2. User Profile Management

### User Story: Update Profile Details
*   [ ] **Action**: Go to the Profile page, change the **Full Name**, and click **Save All Changes**.
*   [ ] **Verification**: The success toast is shown, and the header dropdown/mobile menu reflects the new name instantly.
*   [ ] **Action**: Enter a new password in the password field and click **Save All Changes**.
*   [ ] **Verification**: Sign out, log back in, and confirm that the new password is required to access the account (verifying database password hashing).

### User Story: Profile Picture Customization
*   [ ] **Action**: On the Profile page, tap the avatar/camera icon, choose a photo, and save.
*   [ ] **Verification**: The photo changes immediately on the profile card, in the top-right header, and in the mobile menu drawer, persisting across page reloads.

---

## 💬 3. Chat & Messaging

### User Story: Contact Support
*   [ ] **Action**: Submit the contact form under **Contact Us** with a Subject and Message.
*   [ ] **Verification**: Success popup is displayed, confirming that the email message was sent to the administrator.

### User Story: Search and Chat with Users
*   [ ] **Action**: Tap **New Conversation** on the Chat screen, type a user's exact username, and search.
*   [ ] **Verification**: Searching for an existing user starts a conversation drawer. Typing a non-existent user returns a clear `"User not found"` message.
*   [ ] **Action**: Send messages back and forth.
*   [ ] **Verification**: Messages are delivered and displayed in real-time.

---

## 🤖 4. AI Chatbot Assistant

### User Story: Conversational Chatbot
*   [ ] **Action**: Navigate to **AI Assistant** and send a prompt (e.g., *"How do I sign hello?"*).
*   [ ] **Verification**: Bouncing dots typing indicator appears, and a clean, context-aware reply is returned from Llama 3.

---

## 📹 5. Video Calls & Sign Language Translation

### User Story: Real-time Camera Translation
*   [ ] **Action**: Open the **Translate** page and grant camera permissions.
*   [ ] **Verification**: Signs performed in front of the camera are captured and transcribed into text on screen.

### User Story: Live Video Call with AI Translation
*   [ ] **Action**: Initiate a video call to another user in the Chat section.
*   [ ] **Verification**: WebRTC connection establishes successfully.
*   [ ] **Action**: Click the **Translate (robot/globe)** button during the call.
*   [ ] **Verification**: AI Translation activates, transcribing sign language gestures in real-time as overlay subtitles on the screen.
