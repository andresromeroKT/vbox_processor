# lambda_handler.py
import json
import logging
from typing import Dict, Any
from functionalities.h3_simplifier import request_processor
from functionalities.requests import validate_reference_code

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for processing VBox data
    
    Expected event format:
    {
        "reference_code": "137319VBOX"
    }
    
    Or via API Gateway:
    {
        "body": "{\"reference_code\": \"137319VBOX\"}"
    }
    """
    
    try:
        # Extract reference_code from event
        reference_code = None
        
        # Handle direct invocation
        if 'reference_code' in event:
            reference_code = event['reference_code']
        
        # Handle API Gateway invocation
        elif 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            reference_code = body.get('reference_code')
        
        # Handle query parameters (API Gateway GET)
        elif 'queryStringParameters' in event and event['queryStringParameters']:
            reference_code = event['queryStringParameters'].get('reference_code')
        
        if not reference_code:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing reference_code parameter',
                    'message': 'Please provide reference_code in the request'
                })
            }
        
        logger.info(f"Processing reference_code: {reference_code}")
        
        # Check if reference already exists in processed table
        if validate_reference_code(reference_code):
            logger.info(f"Reference {reference_code} already processed, skipping")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Reference {reference_code} already processed',
                    'status': 'skipped',
                    'reference_code': reference_code
                })
            }
        
        # Process the reference
        request_processor(reference_code)
        
        logger.info(f"Successfully processed reference: {reference_code}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed reference {reference_code}',
                'status': 'completed',
                'reference_code': reference_code
            })
        }
        
    except Exception as e:
        error_message = f"Error processing reference {reference_code if 'reference_code' in locals() else 'unknown'}: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': error_message,
                'status': 'failed',
                'reference_code': reference_code if 'reference_code' in locals() else None
            })
        }

# For local testing
if __name__ == '__main__':
    # Test event
    test_event = {
        'reference_code': '137319VBOX'
    }
    
    class MockContext:
        def __init__(self):
            self.function_name = 'test-function'
            self.memory_limit_in_mb = 512
            self.remaining_time_in_millis = 30000
    
    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2))