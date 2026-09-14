import { Card, CardContent, Typography } from "@mui/material";

export default function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <Card variant="outlined" sx={{ minWidth: 160, flex: 1 }}>
      <CardContent>
        <Typography variant="overline" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="h5">{value}</Typography>
      </CardContent>
    </Card>
  );
}
