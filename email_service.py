# email_service.py
import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_prediction_email(recipient_email, symbol, predictions, analysis):
    """
    Send prediction results via email
    """
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    
    # Debug logging
    logger.info(f"Attempting to send email for {symbol}")
    logger.info(f"EMAIL_SENDER configured: {'Yes' if sender_email else 'No'}")
    logger.info(f"EMAIL_PASSWORD configured: {'Yes' if sender_password else 'No'}")
    
    if not sender_email or not sender_password:
        logger.error("Email not configured. Set EMAIL_SENDER and EMAIL_PASSWORD in environment variables")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"🚀 AlphaAnalytics AI Prediction Report for {symbol}"
        
        # Create HTML version
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .content {{ background: #f9fafb; padding: 20px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 10px 10px; }}
                .price-box {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #6366f1; }}
                .price {{ font-size: 20px; font-weight: bold; color: #111827; }}
                .label {{ color: #6b7280; font-size: 14px; }}
                .trend-up {{ color: #10b981; }}
                .trend-down {{ color: #ef4444; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #9ca3af; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 AlphaAnalytics AI Prediction</h1>
                    <p style="margin: 5px 0 0; opacity: 0.9;">Powered by Gemini AI</p>
                </div>
                <div class="content">
                    <h2 style="color: #6366f1; margin-top: 0;">{symbol}</h2>
                    
                    <div class="price-box">
                        <h3 style="margin-top: 0;">📈 Price Predictions</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td class="label">Open:</td>
                                <td class="price">${predictions['open']['value']}</td>
                                <td style="color: #6b7280;">(Conf: {predictions['open']['confidence']}%)</td>
                            </tr>
                            <tr>
                                <td class="label">High:</td>
                                <td class="price">${predictions['high']['value']}</td>
                                <td style="color: #6b7280;">(Conf: {predictions['high']['confidence']}%)</td>
                            </tr>
                            <tr>
                                <td class="label">Low:</td>
                                <td class="price">${predictions['low']['value']}</td>
                                <td style="color: #6b7280;">(Conf: {predictions['low']['confidence']}%)</td>
                            </tr>
                            <tr>
                                <td class="label">Close:</td>
                                <td class="price">${predictions['close']['value']}</td>
                                <td style="color: #6b7280;">(Conf: {predictions['close']['confidence']}%)</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="price-box">
                        <h3 style="margin-top: 0;">📊 Trend Analysis</h3>
                        <p><strong>Trend:</strong> 
                            <span class="{'trend-up' if predictions.get('trend') == 'BULLISH' else 'trend-down' if predictions.get('trend') == 'BEARISH' else ''}">
                                {predictions.get('trend', 'NEUTRAL')}
                            </span>
                        </p>
                        <p><strong>Strength:</strong> {predictions.get('trend_strength', 50)}%</p>
                        <p><strong>Recommendation:</strong> 
                            <span style="font-weight: bold; color: {'#10b981' if predictions.get('recommendation') in ['BUY', 'STRONG BUY'] else '#ef4444' if predictions.get('recommendation') in ['SELL', 'STRONG SELL'] else '#f59e0b'}">
                                {predictions.get('recommendation', 'HOLD')}
                            </span>
                        </p>
                    </div>
                    
                    <div class="price-box">
                        <h3 style="margin-top: 0;">🤖 AI Analysis</h3>
                        <p style="font-style: italic;">"{analysis}"</p>
                    </div>
                    
                    <p style="color: #6b7280; font-size: 14px; margin-top: 20px;">
                        Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
                <div class="footer">
                    <p>© 2026 AlphaAnalytics. All rights reserved.</p>
                    <p>This is an automated AI-generated prediction. Always do your own research.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version (fallback)
        text_body = f"""
        AlphaAnalytics AI Prediction for {symbol}
        
        PRICE PREDICTIONS:
        - Open: ${predictions['open']['value']} (Confidence: {predictions['open']['confidence']}%)
        - High: ${predictions['high']['value']} (Confidence: {predictions['high']['confidence']}%)
        - Low: ${predictions['low']['value']} (Confidence: {predictions['low']['confidence']}%)
        - Close: ${predictions['close']['value']} (Confidence: {predictions['close']['confidence']}%)
        
        TREND: {predictions.get('trend', 'NEUTRAL')} (Strength: {predictions.get('trend_strength', 50)}%)
        RECOMMENDATION: {predictions.get('recommendation', 'HOLD')}
        
        AI ANALYSIS:
        {analysis}
        
        Generated by AlphaAnalytics
        """
        
        # Attach both versions
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        # Connect to Gmail SMTP
        logger.info("Connecting to Gmail SMTP server...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.set_debuglevel(1)  # Enable debug output
        server.starttls()
        
        # Login
        logger.info("Attempting login...")
        server.login(sender_email, sender_password)
        logger.info("Login successful!")
        
        # Send email
        logger.info(f"Sending email to {recipient_email}...")
        server.send_message(msg)
        logger.info("Email sent successfully!")
        
        # Close connection
        server.quit()
        
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication Error: {e}")
        logger.error("This usually means:")
        logger.error("1. You're using your regular Gmail password instead of an App Password")
        logger.error("2. 2-Factor Authentication is enabled and you need an App Password")
        logger.error("3. Less secure app access is blocked by Google")
        return False
        
    except smtplib.SMTPException as e:
        logger.error(f"SMTP Error: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected email error: {e}")
        return False