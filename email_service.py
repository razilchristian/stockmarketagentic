# email_service.py
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_prediction_email(recipient_email, symbol, predictions, analysis):
    """
    Send prediction results via email
    """
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    
    if not sender_email or not sender_password:
        print("❌ Email not configured. Set EMAIL_SENDER and EMAIL_PASSWORD in .env")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"🚀 AlphaAnalytics AI Prediction for {symbol}"
        
        # Email body with nice formatting
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Inter', Arial, sans-serif; background: #0f172a; color: #f8fafc; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 10px; }}
                .content {{ padding: 20px; }}
                .prediction-box {{ background: #1e293b; border-radius: 8px; padding: 15px; margin: 10px 0; }}
                .price {{ font-size: 24px; font-weight: bold; color: #6366f1; }}
                .range {{ color: #94a3b8; font-size: 14px; }}
                .footer {{ text-align: center; color: #64748b; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="color: white;">AlphaAnalytics</h1>
                    <p style="color: rgba(255,255,255,0.9);">AI-Powered Stock Predictions</p>
                </div>
                
                <div class="content">
                    <h2 style="color: #6366f1;">📈 {symbol} Prediction Report</h2>
                    
                    <div class="prediction-box">
                        <h3>📊 Price Predictions for Tomorrow</h3>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div>
                                <p><strong>Open:</strong></p>
                                <p class="price">${predictions['open']['value']}</p>
                                <p class="range">${predictions['open']['lower_bound']} - ${predictions['open']['upper_bound']}</p>
                            </div>
                            
                            <div>
                                <p><strong>Close:</strong></p>
                                <p class="price">${predictions['close']['value']}</p>
                                <p class="range">${predictions['close']['lower_bound']} - ${predictions['close']['upper_bound']}</p>
                            </div>
                            
                            <div>
                                <p><strong>High:</strong></p>
                                <p class="price">${predictions['high']['value']}</p>
                                <p class="range">${predictions['high']['lower_bound']} - ${predictions['high']['upper_bound']}</p>
                            </div>
                            
                            <div>
                                <p><strong>Low:</strong></p>
                                <p class="price">${predictions['low']['value']}</p>
                                <p class="range">${predictions['low']['lower_bound']} - ${predictions['low']['upper_bound']}</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="prediction-box">
                        <h3>🤖 AI Analysis</h3>
                        <p>{analysis}</p>
                    </div>
                    
                    <div style="text-align: center; margin: 20px 0;">
                        <a href="https://stockmarketagentic.onrender.com" style="background: #6366f1; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px;">View Full Dashboard</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p>⚠️ This is an automated AI prediction - not financial advice</p>
                    <p>© 2026 AlphaAnalytics. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to Gmail SMTP
        print("📧 Sending email...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False