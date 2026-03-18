import asyncio, logging, os, time
logger = logging.getLogger(__name__)

PRIVATE_KEY         = os.getenv("POLY_PRIVATE_KEY", "")
RELAYER_KEY         = os.getenv("POLY_RELAYER_API_KEY", "")
RELAYER_SECRET      = os.getenv("POLY_RELAYER_API_SECRET", "")
RELAYER_PASSPHRASE  = os.getenv("POLY_RELAYER_API_PASSPHRASE", "")
PROXY_ADDRESS       = os.getenv("POLY_PROXY_ADDRESS", "")
CLOB_HOST           = "https://clob.polymarket.com"
CHAIN_ID            = 137
_client             = None
_lock               = asyncio.Lock()

async def _get_client():
    global _client
    if _client is not None:
        return _client
    async with _lock:
        if _client is not None:
            return _client
        if not PRIVATE_KEY:
            logger.error("POLY_PRIVATE_KEY not set")
            return None
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            c = await asyncio.to_thread(
                ClobClient, CLOB_HOST,
                key=PRIVATE_KEY,
                chain_id=CHAIN_ID,
                signature_type=2,
                funder=PROXY_ADDRESS or None,
            )
            if RELAYER_KEY:
                creds = ApiCreds(
                    api_key=RELAYER_KEY,
                    api_secret=RELAYER_SECRET,
                    api_passphrase=RELAYER_PASSPHRASE,
                )
                await asyncio.to_thread(c.set_api_creds, creds)
                logger.info("Using relayer API creds: key=%s", RELAYER_KEY)
            else:
                try:
                    creds = await asyncio.to_thread(c.derive_api_key)
                    await asyncio.to_thread(c.set_api_creds, creds)
                    logger.info("API creds derived: key=%s", creds.api_key)
                except Exception as e:
                    logger.error("derive_api_key failed: %s", e)
                    return None
            _client = c
            logger.info("ClobClient ready")
        except Exception as e:
            logger.error("ClobClient init failed: %s", e)
            return None
    return _client

async def place_order(token_id: str, side: str, price: float, size: float):
    if round(size, 2) < 5.0:
        logger.debug("Skipping order: size %.2f shares below minimum 5", size)
        return None
    client = await _get_client()
    if not client:
        return None
    try:
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY, SELL
        args = OrderArgs(
            token_id=token_id,
            price=round(price, 3),
            size=round(size, 2),
            side=BUY if side.upper() == "BUY" else SELL,
            expiration=0,
        )
        logger.info("placing order: %s %s token=%s price=%.3f shares=%.2f maker=%s",
                    side.upper(), token_id[:12], token_id[:12], round(price, 3), round(size, 2),
                    PROXY_ADDRESS[:10] if PROXY_ADDRESS else "EOA")
        signed = await asyncio.to_thread(client.create_order, args)
        result = await asyncio.to_thread(client.post_order, signed)
        logger.info("LIVE ORDER: %s %s price=%.3f size=$%.2f → %s",
                    side.upper(), token_id[:12], price, size, result)
        return result
    except Exception as e:
        error_msg = str(e)
        logger.error("Order failed: %s", e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            global _client
            _client = None
            logger.info("Credentials expired — will reinitialize on next order")
        return None

async def _run_startup_allowance():
    client = await _get_client()
    if not client:
        return

    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        # Force CLOB to refresh its internal balance/allowance state from on-chain
        for sig_type in (0, 1, 2):
            try:
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig_type)
                updated = await asyncio.to_thread(client.update_balance_allowance, params)
                logger.info("update_balance_allowance sig_type=%d: %s", sig_type, updated)
            except Exception as e:
                logger.error("update_balance_allowance sig_type=%d failed: %s", sig_type, e)
        # Read final state
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
            bal = await asyncio.to_thread(client.get_balance_allowance, params)
            logger.info("balance after update sig_type=2: %s", bal)
        except Exception as e:
            logger.error("get_balance_allowance failed: %s", e)
    except ImportError:
        logger.warning("BalanceAllowanceParams not available in this py-clob-client version")

async def cancel_all():
    client = await _get_client()
    if not client:
        return []
    try:
        result = await asyncio.to_thread(client.cancel_all)
        logger.info("cancel_all: %s", result)
        return result or []
    except Exception as e:
        logger.error("cancel_all failed: %s", e)
        return []

CLOB_READY = bool(PRIVATE_KEY)
