import asyncio
from datetime import datetime, timedelta
import pytz
from logging_utils import banknifty_logger as logger
from bankNifty_config import Config
import pandas as pd

ist = pytz.timezone('Asia/Kolkata')

class SystemTransitionManager:
    def __init__(self, old_system, new_system):
        self.old_system = old_system
        self.new_system = new_system
        self.transition_state = 'preparation'
        self.parallel_running = False
        self.comparison_metrics = []
        self.transition_complete = False

    async def start_transition(self):
        """Start the transition process"""
        try:
            logger.info("Starting system transition process...")
            
            # Phase 1: Preparation
            if not await self._prepare_transition():
                return False

            # Phase 2: Parallel Running
            if not await self._run_parallel_systems():
                return False

            # Phase 3: Validation
            if not await self._validate_new_system():
                return False

            # Phase 4: Switchover
            if not await self._perform_switchover():
                return False

            logger.info("System transition completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in transition process: {e}")
            logger.exception("Full traceback:")
            return False

    async def _prepare_transition(self):
        """Prepare for transition"""
        try:
            logger.info("Preparing for system transition...")
            
            # Check current positions
            current_positions = self.old_system.position_manager.get_positions()
            if current_positions:
                logger.warning(f"Active positions found: {len(current_positions)}. Waiting for positions to close...")
                return False

            # Initialize new system
            if not await self._initialize_new_system():
                return False

            # Verify data consistency
            if not self._verify_data_consistency():
                return False

            self.transition_state = 'ready'
            return True

        except Exception as e:
            logger.error(f"Error in transition preparation: {e}")
            return False

    async def _initialize_new_system(self):
        """Initialize and validate new system"""
        try:
            # Train ML model with historical data
            training_successful = await self.new_system.ml_model.train(
                historical_data=await self._get_training_data(),
                options_data=await self._get_options_data(),
                market_conditions=self._get_market_conditions(),
                sentiment_data=await self._get_sentiment_data()
            )

            if not training_successful:
                logger.error("Failed to train new ML model")
                return False

            # Verify model performance
            validation_result = await self._validate_ml_model()
            if not validation_result['success']:
                logger.error(f"ML model validation failed: {validation_result['reason']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error initializing new system: {e}")
            return False

    async def _run_parallel_systems(self):
        """Run both systems in parallel for comparison"""
        try:
            self.parallel_running = True
            comparison_period = timedelta(days=Config.TRANSITION_COMPARISON_DAYS)
            start_time = datetime.now(ist)

            while datetime.now(ist) - start_time < comparison_period:
                # Get market data
                market_data = await self._get_market_data()
                
                # Get predictions from both systems
                old_prediction = await self._get_old_system_prediction(market_data)
                new_prediction = await self._get_new_system_prediction(market_data)
                
                # Compare predictions
                comparison = self._compare_predictions(old_prediction, new_prediction)
                self.comparison_metrics.append(comparison)

                # Check for significant divergence
                if self._check_divergence(comparison):
                    logger.warning("Significant divergence detected between systems")
                    await self._handle_divergence(comparison)

                await asyncio.sleep(Config.COMPARISON_INTERVAL)

            return self._analyze_comparison_results()

        except Exception as e:
            logger.error(f"Error in parallel running phase: {e}")
            return False

    async def _validate_new_system(self):
        """Validate new system performance"""
        try:
            metrics = self._calculate_validation_metrics()
            
            validation_criteria = {
                'accuracy_threshold': 0.60,
                'consistency_threshold': 0.75,
                'risk_compliance': True
            }

            validation_results = {
                'accuracy': metrics['prediction_accuracy'] > validation_criteria['accuracy_threshold'],
                'consistency': metrics['signal_consistency'] > validation_criteria['consistency_threshold'],
                'risk_compliance': self._verify_risk_compliance()
            }

            if all(validation_results.values()):
                logger.info("New system validation successful")
                return True
            else:
                logger.error(f"System validation failed: {validation_results}")
                return False

        except Exception as e:
            logger.error(f"Error in system validation: {e}")
            return False

    async def _perform_switchover(self):
        """Perform the actual system switchover"""
        try:
            logger.info("Initiating system switchover...")
            
            # Verify no active trades
            if not self._verify_no_active_trades():
                return False

            # Backup old system state
            self._backup_system_state()

            # Switch to new system
            if not await self._switch_systems():
                return False

            # Verify new system operation
            if not await self._verify_new_system_operation():
                return False

            self.transition_complete = True
            logger.info("System switchover completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during system switchover: {e}")
            return False

    def _verify_risk_compliance(self):
        """Verify risk management compliance of new system"""
        try:
            risk_checks = {
                'position_sizing': self._check_position_sizing(),
                'stop_loss_compliance': self._check_stop_loss_compliance(),
                'exposure_limits': self._check_exposure_limits()
            }
            
            return all(risk_checks.values())
        except Exception as e:
            logger.error(f"Error in risk compliance verification: {e}")
            return False

    def get_transition_status(self):
        """Get current transition status"""
        return {
            'state': self.transition_state,
            'parallel_running': self.parallel_running,
            'metrics': self._get_transition_metrics(),
            'completion': self.transition_complete
        }

    def _get_transition_metrics(self):
        """Calculate transition metrics"""
        if not self.comparison_metrics:
            return {}
            
        return {
            'signal_agreement_rate': self._calculate_agreement_rate(),
            'risk_compliance_rate': self._calculate_risk_compliance_rate(),
            'performance_comparison': self._compare_system_performance()
        }