import smtplib
from email.mime.text import MIMEText
from config.settings import settings

def send_email(to_email: str, subject: str, body: str):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = settings.SMTP_EMAIL
    msg['To'] = to_email

    try:
        with smtplib.SMTP(settings.SMTP_SERVER,settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_EMAIL,settings.SMTP_APP_PASSWORD)
            server.sendmail(
                settings.SMTP_EMAIL,  # from
                to_email,             # to
                msg.as_string()       # message content
            )
            print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}") 
        raise e 