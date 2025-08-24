import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import json
from datetime import datetime
import pytz
from pathlib import Path

ist = pytz.timezone('Asia/Kolkata')

class NotificationManager:
    def __init__(self):
        self.notification_history = []
        self.email_settings = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': Config.EMAIL_USERNAME,
            'password': Config.EMAIL_PASSWORD
        }

    def send_trade_notification(self, trade_data):
        """Send notification for trade execution"""
        try:
            subject = f"Trade Alert: {trade_data['type']} {trade_data['symbol']}"
            message = self._format_trade_message(trade_data)
            
            if Config.ENABLE_EMAIL_NOTIFICATIONS:
                self._send_email(subject, message)
            
            self._log_notification({
                'type': 'trade',
                'data': trade_data,
                'timestamp': datetime.now(ist)
            })

        except Exception as e:
            logger.error(f"Error sending trade notification: {e}")

    def send_alert(self, alert_type, alert_data):
        """Send general alert notification"""
        try:
            subject = f"Alert: {alert_type}"
            message = self._format_alert_message(alert_type, alert_data)
            
            if Config.ENABLE_EMAIL_NOTIFICATIONS:
                self._send_email(subject, message)
            
            self._log_notification({
                'type': 'alert',
                'alert_type': alert_type,
                'data': alert_data,
                'timestamp': datetime.now(ist)
            })

        except Exception as e:
            logger.error(f"Error sending alert: {e}")

    def send_performance_update(self, performance_data):
        """Send performance update notification"""
        try:
            subject = "Strategy Performance Update"
            message = self._format_performance_message(performance_data)
            
            if Config.ENABLE_EMAIL_NOTIFICATIONS:
                self._send_email(subject, message)
            
            self._log_notification({
                'type': 'performance',
                'data': performance_data,
                'timestamp': datetime.now(ist)
            })

        except Exception as e:
            logger.error(f"Error sending performance update: {e}")

    def _send_email(self, subject, message):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.email_settings['username']
            msg['To'] = Config.EMAIL_RECIPIENT

            msg.attach(MIMEText(message, 'plain'))

            with smtplib.SMTP(self.email_settings['smtp_server'], self.email_settings['smtp_port']) as server:
                server.starttls()
                server.login(self.email_settings['username'], self.email_settings['password'])
                server.send_message(msg)

            logger.info(f"Email sent: {subject}")

        except Exception as e:
            logger.error(f"Error sending email: {e}")

    def _format_trade_message(self, trade_data):
        """Format trade notification message"""
        return f"""
Trade Details:
-------------
Symbol: {trade_data['symbol']}
Type: {trade_data['type']}
Quantity: {trade_data['quantity']}
Price: {trade_data['price']}
Time: {trade_data['timestamp']}
"""

    def _format_alert_message(self, alert_type, alert_data):
        """Format alert message"""
        return f"""
Alert Type: {alert_type}
-------------
{json.dumps(alert_data, indent=2)}
Time: {datetime.now(ist)}
"""

    def _format_performance_message(self, performance_data):
        """Format performance update message"""
        return f"""
Performance Update:
-----------------
Total Trades: {performance_data['total_trades']}
Win Rate: {performance_data['win_rate']:.2f}%
Profit Factor: {performance_data['profit_factor']:.2f}
Total P&L: {performance_data['total_pnl']:.2f}
Sharpe Ratio: {performance_data['sharpe_ratio']:.2f}
Max Drawdown: {performance_data['max_drawdown']:.2f}
Time: {datetime.now(ist)}
"""

    def _log_notification(self, notification_data):
        """Log notification to history"""
        try:
            self.notification_history.append(notification_data)
            
            # Save to file
            notifications_file = Path(Config.DATA_DIR) / 'notifications.json'
            notifications_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(notifications_file, 'w') as f:
                json.dump(self.notification_history, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error logging notification: {e}")