import smtplib
from email.mime.text import MIMEText
from config.settings import settings

def send_email(to_email: str, subject: str, body: str) -> bool:
    """
    Send an email. Returns True on success, False on failure.
    Never raises — callers must check the return value.
    Raising SMTP exceptions after a DB commit corrupts the session state.
    """
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = settings.SMTP_EMAIL
    msg['To'] = to_email

    try:
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_EMAIL, settings.SMTP_APP_PASSWORD)
            server.sendmail(
                settings.SMTP_EMAIL,
                to_email,
                msg.as_string()
            )
            print(f"Email sent successfully to {to_email}")
            return True
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")
        return False